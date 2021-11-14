from timeit import default_timer as timer 
def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    if device:  
        print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    else:
        print(f"\nTrain time: {total_time:.3f} seconds")
    return round(total_time, 3)
