import tkinter as tk
import numpy as np


class PixelatedPainter:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixelated Painter")

        # Size of the pixel grid
        self.grid_size = 28

        # Create a canvas
        self.canvas = tk.Canvas(root, width=700, height=700, bg="white")
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        # Initialize the pixels array
        self.pixels = [[0] * self.grid_size for _ in range(self.grid_size)]

        # Set up mouse bindings
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def paint(self, event):
        # Get the current mouse position
        x, y = event.x, event.y

        # Map the mouse position to the grid
        grid_x = x // (700 // self.grid_size)
        grid_y = y // (700 // self.grid_size)

        # Paint the grid cell
        self.canvas.create_rectangle(
            grid_x * (700 // self.grid_size),
            grid_y * (700 // self.grid_size),
            (grid_x + 1) * (700 // self.grid_size),
            (grid_y + 1) * (700 // self.grid_size),
            fill="black",
            outline="black"
        )

        # Update the pixels array
        self.pixels[grid_y][grid_x] = 1

        print(np.shape(self.pixels))

    def reset(self, event):
        # Reset the drawing position when the mouse is released
        self.last_x = None
        self.last_y = None

if __name__ == "__main__":
    root = tk.Tk()
    app = PixelatedPainter(root)
    root.mainloop()