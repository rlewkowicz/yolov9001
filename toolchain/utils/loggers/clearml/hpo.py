from clearml import Task

from clearml.automation import HyperParameterOptimizer, UniformParameterRange
from clearml.automation.optuna import OptimizerOptuna

task = Task.init(project_name='Hyper-Parameter Optimization',
                 task_name='YOLOv5',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

optimizer = HyperParameterOptimizer(

    base_task_id='<your_template_task_id>',

    hyper_parameters=[
        UniformParameterRange('Hyperparameters/lr0', min_value=1e-5, max_value=1e-1),
        UniformParameterRange('Hyperparameters/lrf', min_value=0.01, max_value=1.0),
        UniformParameterRange('Hyperparameters/momentum', min_value=0.6, max_value=0.98),
        UniformParameterRange('Hyperparameters/weight_decay', min_value=0.0, max_value=0.001),
        UniformParameterRange('Hyperparameters/warmup_epochs', min_value=0.0, max_value=5.0),
        UniformParameterRange('Hyperparameters/warmup_momentum', min_value=0.0, max_value=0.95),
        UniformParameterRange('Hyperparameters/warmup_bias_lr', min_value=0.0, max_value=0.2),
        UniformParameterRange('Hyperparameters/box', min_value=0.02, max_value=0.2),
        UniformParameterRange('Hyperparameters/cls', min_value=0.2, max_value=4.0),
        UniformParameterRange('Hyperparameters/cls_pw', min_value=0.5, max_value=2.0),
        UniformParameterRange('Hyperparameters/obj', min_value=0.2, max_value=4.0),
        UniformParameterRange('Hyperparameters/obj_pw', min_value=0.5, max_value=2.0),
        UniformParameterRange('Hyperparameters/iou_t', min_value=0.1, max_value=0.7),
        UniformParameterRange('Hyperparameters/anchor_t', min_value=2.0, max_value=8.0),
        UniformParameterRange('Hyperparameters/fl_gamma', min_value=0.0, max_value=4.0),
        UniformParameterRange('Hyperparameters/hsv_h', min_value=0.0, max_value=0.1),
        UniformParameterRange('Hyperparameters/hsv_s', min_value=0.0, max_value=0.9),
        UniformParameterRange('Hyperparameters/hsv_v', min_value=0.0, max_value=0.9),
        UniformParameterRange('Hyperparameters/degrees', min_value=0.0, max_value=45.0),
        UniformParameterRange('Hyperparameters/translate', min_value=0.0, max_value=0.9),
        UniformParameterRange('Hyperparameters/scale', min_value=0.0, max_value=0.9),
        UniformParameterRange('Hyperparameters/shear', min_value=0.0, max_value=10.0),
        UniformParameterRange('Hyperparameters/perspective', min_value=0.0, max_value=0.001),
        UniformParameterRange('Hyperparameters/flipud', min_value=0.0, max_value=1.0),
        UniformParameterRange('Hyperparameters/fliplr', min_value=0.0, max_value=1.0),
        UniformParameterRange('Hyperparameters/mosaic', min_value=0.0, max_value=1.0),
        UniformParameterRange('Hyperparameters/mixup', min_value=0.0, max_value=1.0),
        UniformParameterRange('Hyperparameters/copy_paste', min_value=0.0, max_value=1.0)],

    objective_metric_title='metrics',
    objective_metric_series='mAP_0.5',

    objective_metric_sign='max',

    max_number_of_concurrent_tasks=1,

    optimizer_class=OptimizerOptuna,

    save_top_k_tasks_only=5,
    compute_time_limit=None,
    total_max_jobs=20,
    min_iteration_per_job=None,
    max_iteration_per_job=None,
)

optimizer.set_report_period(10 / 60)

optimizer.set_time_limit(in_minutes=120.0)

optimizer.start_locally()

optimizer.wait()

optimizer.stop()

print('We are done, good bye')
