import pycuda.driver as cuda
import pycuda.autoinit

def main():
    # Initialize the CUDA driver
    cuda.init()

    # Get the number of CUDA devices
    device_count = cuda.Device.count()
    
    print(f"Number of CUDA devices available: {device_count}")

    # Optionally, print information about each device
    for i in range(device_count):
        device = cuda.Device(i)
        device_name = device.name()
        compute_capability = device.compute_capability()
        total_memory = device.total_memory() // (1024**2)  # Convert bytes to MB
        print(f"Device {i}: {device_name}")
        print(f"  Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
        print(f"  Total Memory: {total_memory} MB")

if __name__ == "__main__":
    main()
