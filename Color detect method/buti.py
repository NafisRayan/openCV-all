import turtle
import random

def draw_flame():
    # Set up turtle
    screen = turtle.Screen()
    screen.setup(800, 600)
    screen.title("Flame")
    screen.bgcolor("black")
    screen.tracer(0)
    
    # Create turtle object
    flame = turtle.Turtle()
    flame.shape("circle")
    flame.shapesize(1)
    flame.width(3)
    flame.speed(0)
    
    while True:
        # Set random flame color
        colors = ["#FF6633", "#FF9966", "#FFCC33", "#FF3300", "#FF5050"]
        flame.color(random.choice(colors))
        
        # Calculate flame size and position
        size = random.randint(20, 100)
        x = random.randint(-400, 400)
        y = random.randint(-300, 300)
        
        # Draw flame
        flame.penup()
        flame.goto(x, y)
        flame.pendown()
        flame.begin_fill()
        for _ in range(size):
            flame.forward(1)
            flame.left(1)
        flame.end_fill()
        
        # Refresh screen
        screen.update()

    turtle.done()

# Call the function to draw the flame
draw_flame()
