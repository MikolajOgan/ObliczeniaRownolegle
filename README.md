# ParallelCcomputing

Distributed Image Blur (Python)
Simple distributed image-blurring demo that splits an image into vertical fragments, sends them to one or more remote workers, and stitches the processed fragments back into a final blurred image.
​

Files
client.py — A TCP worker (server) that receives an image fragment (NumPy array) over a socket, applies a CPU Gaussian blur, and sends the blurred fragment back.
​

server3.py — A controller (client) that loads an input image, splits it into parts_count fragments, dispatches fragments to multiple servers in parallel threads (TCP or HTTP), then merges results and writes blurred_image.png.
​
