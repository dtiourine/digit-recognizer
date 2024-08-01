import random
import typer
from matplotlib import pyplot as plt
from modeling.load_data import load_data

app = typer.Typer()

# Load the training data
train_data, _, _, _ = load_data(transform=None)

def show_random_image(event=None):
    # Generate a random index within the range of the training dataset
    random_index = random.randint(0, len(train_data) - 1)

    # Retrieve the image and label corresponding to the random index
    image, label = train_data[random_index]

    # Clear the previous plot
    plt.clf()

    # Display the image
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Label: {label}")
    plt.axis('off')  # Hide the axes for a cleaner display
    plt.draw()  # Redraw the figure with the new image

@app.command()
def main():
    # Set up the plot
    fig, ax = plt.subplots()

    # Display the first random image
    show_random_image()

    # Connect the figure to key press events
    fig.canvas.mpl_connect('key_press_event', show_random_image)

    # Show the figure with interactive mode on
    plt.show()

if __name__ == "__main__":
    app()
