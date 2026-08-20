from __future__ import annotations

import math
import random
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from flbench.algorithms.registry import build_algorithm
from flbench.data.datasets import build_dataset
from flbench.metrics.evaluation import evaluate
from flbench.models.factory import build_model
from flbench.seeding import seed_everything
from .logging import CSVLogger, save_json
from .naming import experiment_id
from .resources import EnergyTracker, communication_metrics, system_metadata
from .stopping import StoppingController


ROUND_FIELDS = [
    'experiment_id','dataset','algorithm','round','seed','trial','alpha',
    'num_clients','selected_clients','participation_rate','local_epochs',
    'batch_size','learning_rate','optimizer','evaluated','mean_local_loss',
    'val_loss','val_accuracy','test_loss','test_accuracy','macro_f1',
    'client_compute_time_sum_sec','client_compute_time_max_sec',
    'server_aggregation_time_sec','evaluation_time_sec','round_wall_time_sec',
    'cumulative_wall_time_sec','download_bytes','upload_bytes','total_comm_bytes',
    'modeled_comm_time_sequential_sec','modeled_comm_time_parallel_sec',
    'modeled_comm_time_sec','modeled_comm_energy_j','client_cpu_energy_measured_j',
    'client_gpu_energy_measured_j','client_compute_energy_modeled_j',
    'client_compute_energy_reported_j','server_cpu_energy_measured_j',
    'server_gpu_energy_measured_j','server_compute_energy_modeled_j',
    'server_compute_energy_reported_j','evaluation_cpu_energy_measured_j',
    'evaluation_gpu_energy_measured_j','evaluation_compute_energy_modeled_j',
    'evaluation_compute_energy_reported_j','round_cpu_energy_measured_j',
    'round_gpu_energy_measured_j','round_compute_energy_modeled_j',
    'round_compute_energy_reported_j','round_total_energy_reported_j',
    'energy_source','energy_is_estimated','stopping_reason'
]

CLIENT_FIELDS = [
    'experiment_id','algorithm','round','client_id','num_samples','local_loss',
    'local_steps','compute_time_sec','upload_bytes','cpu_energy_measured_j',
    'gpu_energy_measured_j','compute_energy_modeled_j','compute_energy_reported_j',
    'cpu_utilization_fraction','energy_source','energy_is_estimated'
]


def device_of(value):
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(value)


def make_loader(dataset, batch, shuffle, workers, generator):
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator,
    )


def _optional_sum(values):
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _energy_sources(readings):
    sources = sorted({
        reading.energy_source
        for reading in readings
        if reading.energy_source != 'unavailable'
    })
    return '|'.join(sources) if sources else 'unavailable'


def _latest_checkpoint(output_dir):
    files = sorted(output_dir.glob('checkpoint_round_*.pt'))
    return files[-1] if files else None


def _best_model_path(output_dir: Path) -> Path:
    """Checkpoint containing the model with the best observed validation score."""
    return output_dir / 'best_validation_model.pt'


def run_experiment(cfg):
    """Run one FL experiment and return its output directory.

    Convergence-based runs are monitored using validation accuracy.  Whenever
    validation improves by more than ``stopping.min_delta``, the corresponding
    global model is saved.  At termination we restore that best-validation
    model and perform the final validation/test evaluation.  Consequently,
    confirmation accuracy is measured at the best observed validation point,
    not at an arbitrary later round after the metric has already plateaued.
    """
    seed = int(cfg['experiment']['seed'])
    seed_everything(seed, bool(cfg['experiment'].get('deterministic', True)))
    eid = experiment_id(cfg)

    configured_output = Path(cfg['experiment']['output_dir'])
    output_dir = (
        configured_output
        if bool(cfg['experiment'].get('use_exact_output_dir', False))
        else configured_output / eid
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(cfg, output_dir / 'resolved_config.json')
    save_json({'status': 'running', 'experiment_id': eid}, output_dir / 'status.json')

    device = device_of(cfg['experiment']['device'])
    save_json(system_metadata(device), output_dir / 'system_metadata.json')

    try:
        # ------------------------ Data and model setup ------------------------
        bundle = build_dataset(cfg)
        save_json({
            'dataset': cfg['data']['dataset'],
            'task_type': bundle.task_type,
            'num_classes': bundle.num_classes,
            'input_shape': bundle.input_shape,
            'num_clients': len(bundle.client_indices),
            'client_names': bundle.client_names,
            'metadata': bundle.metadata,
        }, output_dir / 'partition_metadata.json')

        batch = int(cfg['training']['batch_size'])
        workers = int(cfg['training'].get('num_workers', 0))
        client_generators = {
            cid: torch.Generator().manual_seed(seed + cid)
            for cid in bundle.client_indices
        }
        client_loaders = {
            cid: make_loader(
                Subset(bundle.train_dataset, indices),
                batch,
                True,
                workers,
                client_generators[cid],
            )
            for cid, indices in bundle.client_indices.items()
        }

        validation_loader = make_loader(
            bundle.validation_dataset,
            batch,
            False,
            workers,
            torch.Generator().manual_seed(seed),
        )
        test_loader = make_loader(
            bundle.test_dataset,
            batch,
            False,
            workers,
            torch.Generator().manual_seed(seed),
        )

        torch.manual_seed(int(cfg['model']['initialization_seed']))
        global_model = build_model(cfg, bundle).to(device)
        algorithm = build_algorithm(cfg)
        num_clients = len(bundle.client_indices)
        algorithm.initialize(global_model, num_clients, cfg)

        # -------------------------- Logging / state ---------------------------
        round_logger = CSVLogger(output_dir / 'round_metrics.csv', ROUND_FIELDS)
        client_logger = CSVLogger(output_dir / 'client_metrics.csv', CLIENT_FIELDS)
        stopping = StoppingController(cfg)

        client_rng = random.Random(int(cfg['federation']['client_sampling_seed']))
        participation = float(cfg['federation']['participation_rate'])
        selected_count = max(1, math.ceil(participation * num_clients))

        # For convergence runs this value is a safety cap, not the intended
        # stopping condition.  The normal exit is validation convergence.
        maximum_rounds = int(cfg['federation']['communication_rounds'])
        eval_frequency = int(cfg['evaluation'].get('frequency', 1))

        start_round = 1
        elapsed_before = 0.0
        best_path = _best_model_path(output_dir)

        # ---------------------------- Resume support -------------------------
        if bool(cfg['output'].get('resume', False)):
            checkpoint = _latest_checkpoint(output_dir)
            if checkpoint is not None:
                state = torch.load(checkpoint, map_location=device, weights_only=False)
                global_model.load_state_dict(state['model'])
                algorithm.load_state_dict(state.get('algorithm_state', {}))
                client_rng.setstate(state['client_rng_state'])
                stopping.load_state_dict(state.get('stopping_state', {}))
                elapsed_before = float(state.get('elapsed_sec', 0.0))
                start_round = int(state['round']) + 1
                for cid, generator_state in state.get(
                    'client_loader_generator_states', {}
                ).items():
                    cid = int(cid)
                    if cid in client_generators:
                        client_generators[cid].set_state(generator_state)

        experiment_start = time.perf_counter() - elapsed_before
        final_row = {}
        last_eval = {
            'val_loss': None,
            'val_accuracy': None,
            'test_loss': None,
            'test_accuracy': None,
            'macro_f1': None,
        }

        # ------------------------- Federated training -------------------------
        for round_idx in range(start_round, maximum_rounds + 1):
            round_start = time.perf_counter()
            selected_clients = client_rng.sample(
                list(bundle.client_indices),
                min(selected_count, num_clients),
            )

            updates = []
            client_times = []
            client_energy = []

            # Each selected client trains from the same current global model.
            for client_id in selected_clients:
                tracker = EnergyTracker(cfg, device, phase=f'client_{client_id}')
                tracker.start()
                client_start = time.perf_counter()
                update = algorithm.client_update(
                    client_id,
                    global_model,
                    client_loaders[client_id],
                    device,
                    cfg,
                )
                compute_time = time.perf_counter() - client_start
                energy = tracker.stop()
                upload_bytes = algorithm.client_upload_num_bytes(update)

                client_times.append(compute_time)
                client_energy.append(energy)
                updates.append(update)

                client_logger.append({
                    'experiment_id': eid,
                    'algorithm': algorithm.name,
                    'round': round_idx,
                    'client_id': client_id,
                    'num_samples': update.num_samples,
                    'local_loss': update.metrics.get('local_loss'),
                    'local_steps': update.metrics.get('local_steps'),
                    'compute_time_sec': compute_time,
                    'upload_bytes': upload_bytes,
                    'cpu_energy_measured_j': energy.cpu_energy_measured_j,
                    'gpu_energy_measured_j': energy.gpu_energy_measured_j,
                    'compute_energy_modeled_j': energy.compute_energy_modeled_j,
                    'compute_energy_reported_j': energy.compute_energy_reported_j,
                    'cpu_utilization_fraction': energy.cpu_utilization_fraction,
                    'energy_source': energy.energy_source,
                    'energy_is_estimated': energy.energy_is_estimated,
                })

            # Aggregate the received updates using the selected algorithm.
            server_tracker = EnergyTracker(cfg, device, phase='server_aggregation')
            server_tracker.start()
            aggregation_start = time.perf_counter()
            algorithm.server_update(global_model, updates, cfg)
            aggregation_time = time.perf_counter() - aggregation_start
            server_energy = server_tracker.stop()

            # Validation is intentionally periodic.  Patience is counted only
            # on these evaluation events, so reducing evaluation frequency does
            # not artificially make convergence happen sooner.
            should_evaluate = (
                round_idx % eval_frequency == 0
                or round_idx == maximum_rounds
            )
            evaluation_time = 0.0
            evaluation_energy = None

            if should_evaluate:
                evaluation_tracker = EnergyTracker(cfg, device, phase='evaluation')
                evaluation_tracker.start()
                evaluation_start = time.perf_counter()

                # Validation controls stopping and model selection.  We still
                # log test metrics for backwards compatibility, but the search
                # selector never uses them.  The final reported test metric is
                # recomputed after restoring the best-validation model.
                val = evaluate(global_model, validation_loader, device)
                test = evaluate(global_model, test_loader, device)

                evaluation_time = time.perf_counter() - evaluation_start
                evaluation_energy = evaluation_tracker.stop()
                last_eval = {
                    'val_loss': val['loss'],
                    'val_accuracy': val['accuracy'],
                    'test_loss': test['loss'],
                    'test_accuracy': test['accuracy'],
                    'macro_f1': test['macro_f1'],
                }

            elapsed = time.perf_counter() - experiment_start
            stop_metric = last_eval['val_accuracy'] if should_evaluate else None
            done, reason = stopping.update(
                round_idx,
                stop_metric,
                elapsed,
                maximum_rounds,
            )

            # Persist the strongest validation model immediately when it is
            # observed.  This file is independent of periodic resume checkpoints.
            if should_evaluate and stopping.last_improved:
                torch.save({
                    'round': round_idx,
                    'val_accuracy': last_eval['val_accuracy'],
                    'model': global_model.state_dict(),
                }, best_path)

            # If a non-evaluation round ever triggers another stopping mode,
            # evaluate once before exiting so the run remains fully reported.
            if done and not should_evaluate:
                evaluation_tracker = EnergyTracker(cfg, device, phase='evaluation')
                evaluation_tracker.start()
                evaluation_start = time.perf_counter()
                val = evaluate(global_model, validation_loader, device)
                test = evaluate(global_model, test_loader, device)
                evaluation_time = time.perf_counter() - evaluation_start
                evaluation_energy = evaluation_tracker.stop()
                last_eval = {
                    'val_loss': val['loss'],
                    'val_accuracy': val['accuracy'],
                    'test_loss': test['loss'],
                    'test_accuracy': test['accuracy'],
                    'macro_f1': test['macro_f1'],
                }
                should_evaluate = True

            # Communication accounting is algorithm-aware through the methods
            # below; auxiliary payloads (e.g. SCAFFOLD control variates) are
            # included by the corresponding algorithm implementation.
            download_bytes = algorithm.download_num_bytes(
                global_model,
                len(selected_clients),
            )
            upload_bytes = sum(
                algorithm.client_upload_num_bytes(update)
                for update in updates
            )
            communication = communication_metrics(
                download_bytes=download_bytes,
                upload_bytes=upload_bytes,
                selected_clients=len(selected_clients),
                cfg=cfg,
            )

            all_energy = [*client_energy, server_energy] + (
                [evaluation_energy] if evaluation_energy is not None else []
            )
            client_cpu = _optional_sum(r.cpu_energy_measured_j for r in client_energy)
            client_gpu = _optional_sum(r.gpu_energy_measured_j for r in client_energy)
            client_modeled = _optional_sum(r.compute_energy_modeled_j for r in client_energy)
            client_reported = _optional_sum(r.compute_energy_reported_j for r in client_energy)
            round_cpu = _optional_sum(r.cpu_energy_measured_j for r in all_energy)
            round_gpu = _optional_sum(r.gpu_energy_measured_j for r in all_energy)
            round_modeled = _optional_sum(r.compute_energy_modeled_j for r in all_energy)
            round_reported = _optional_sum(r.compute_energy_reported_j for r in all_energy)
            round_total = (
                None
                if round_reported is None
                else round_reported + communication['modeled_comm_energy_j']
            )

            row = {
                'experiment_id': eid,
                'dataset': cfg['data']['dataset'],
                'algorithm': algorithm.name,
                'round': round_idx,
                'seed': seed,
                'trial': cfg['experiment'].get('trial', 1),
                'alpha': cfg['partition'].get('alpha'),
                'num_clients': num_clients,
                'selected_clients': len(selected_clients),
                'participation_rate': participation,
                'local_epochs': cfg['training']['local_epochs'],
                'batch_size': batch,
                'learning_rate': cfg['training']['learning_rate'],
                'optimizer': cfg['training'].get('optimizer', 'sgd'),
                'evaluated': should_evaluate,
                'mean_local_loss': sum(
                    update.metrics['local_loss'] for update in updates
                ) / len(updates),
                **last_eval,
                'client_compute_time_sum_sec': sum(client_times),
                'client_compute_time_max_sec': max(client_times),
                'server_aggregation_time_sec': aggregation_time,
                'evaluation_time_sec': evaluation_time,
                'round_wall_time_sec': time.perf_counter() - round_start,
                'cumulative_wall_time_sec': elapsed,
                **communication,
                'client_cpu_energy_measured_j': client_cpu,
                'client_gpu_energy_measured_j': client_gpu,
                'client_compute_energy_modeled_j': client_modeled,
                'client_compute_energy_reported_j': client_reported,
                'server_cpu_energy_measured_j': server_energy.cpu_energy_measured_j,
                'server_gpu_energy_measured_j': server_energy.gpu_energy_measured_j,
                'server_compute_energy_modeled_j': server_energy.compute_energy_modeled_j,
                'server_compute_energy_reported_j': server_energy.compute_energy_reported_j,
                'evaluation_cpu_energy_measured_j': None if evaluation_energy is None else evaluation_energy.cpu_energy_measured_j,
                'evaluation_gpu_energy_measured_j': None if evaluation_energy is None else evaluation_energy.gpu_energy_measured_j,
                'evaluation_compute_energy_modeled_j': None if evaluation_energy is None else evaluation_energy.compute_energy_modeled_j,
                'evaluation_compute_energy_reported_j': None if evaluation_energy is None else evaluation_energy.compute_energy_reported_j,
                'round_cpu_energy_measured_j': round_cpu,
                'round_gpu_energy_measured_j': round_gpu,
                'round_compute_energy_modeled_j': round_modeled,
                'round_compute_energy_reported_j': round_reported,
                'round_total_energy_reported_j': round_total,
                'energy_source': _energy_sources(all_energy),
                'energy_is_estimated': any(r.energy_is_estimated for r in all_energy),
                'stopping_reason': reason,
            }
            round_logger.append(row)
            final_row = row

            # Resume checkpoints preserve model, algorithm state, client
            # sampling RNG, DataLoader RNGs and stopping-controller state.
            if cfg['output']['save_checkpoints'] and (
                round_idx % int(cfg['output']['checkpoint_frequency']) == 0
                or done
            ):
                torch.save({
                    'round': round_idx,
                    'model': global_model.state_dict(),
                    'algorithm_state': algorithm.state_dict(),
                    'client_rng_state': client_rng.getstate(),
                    'client_loader_generator_states': {
                        cid: generator.get_state()
                        for cid, generator in client_generators.items()
                    },
                    'stopping_state': stopping.state_dict(),
                    'elapsed_sec': elapsed,
                    'config': cfg,
                }, output_dir / f'checkpoint_round_{round_idx:04d}.pt')

            if done:
                break

        # -------------------- Final best-model confirmation ------------------
        # Restore the best validation model before producing the final summary.
        # This is especially important when patience allows several plateau
        # rounds after the best model was first observed.
        if best_path.exists():
            best_state = torch.load(best_path, map_location=device, weights_only=False)
            global_model.load_state_dict(best_state['model'])
            best_validation_round = int(best_state['round'])
            best_validation_accuracy = float(best_state['val_accuracy'])
        else:
            best_validation_round = int(final_row.get('round', 0))
            best_validation_accuracy = final_row.get('val_accuracy')

        final_val = evaluate(global_model, validation_loader, device)
        final_test = evaluate(global_model, test_loader, device)

        summary = {
            'status': 'completed',
            **final_row,
            # Override these fields so the search selector sees the best
            # validation point and confirmation reports its associated test.
            'val_loss': final_val['loss'],
            'val_accuracy': final_val['accuracy'],
            'test_loss': final_test['loss'],
            'test_accuracy': final_test['accuracy'],
            'macro_f1': final_test['macro_f1'],
            'best_validation_round': best_validation_round,
            'best_validation_accuracy': best_validation_accuracy,
            'termination_round': int(final_row.get('round', 0)),
            'stopping_reason': final_row.get('stopping_reason'),
            'convergence_evaluations_without_improvement': stopping.wait,
        }
        save_json(summary, output_dir / 'summary.json')
        save_json(summary, output_dir / 'status.json')
        return output_dir

    except Exception as exc:
        failure = {
            'status': 'failed',
            'experiment_id': eid,
            'error_type': type(exc).__name__,
            'error': str(exc),
            'traceback': traceback.format_exc(),
        }
        save_json(failure, output_dir / 'status.json')
        raise
