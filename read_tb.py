from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

def read_tensorboard_logs(log_dir):
    # Load the TensorBoard data
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get list of tags (different metrics)
    tags = event_acc.Tags()['scalars']
    
    # Create a dictionary to store all metrics
    data = {}
    
    # Read each metric
    for tag in tags:
        events = event_acc.Scalars(tag)
        data[tag] = pd.DataFrame([
            {'step': event.step, 'value': event.value, 'wall_time': event.wall_time}
            for event in events
        ])
    
    return data

# Usage example:
log_dir = "/local/vondrick/alper/torchtitan-dit-0/outputs/tb_new/dit_20241215-1823"  # You'll need to specify the actual path
metrics = read_tensorboard_logs(log_dir)

# Access specific metrics
loss_data = metrics.get('loss_metrics/global_avg_loss')
mfu_data = metrics.get('mfu(%)')

print(loss_data)
print(mfu_data)
