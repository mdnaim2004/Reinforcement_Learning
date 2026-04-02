import turtle
def draw_squre(side_length):
    for i in range(4):
        turtle.forward(side_length)
        turtle.left(90)

counter = 0

while(counter < 90):
    draw_squre(100)
    turtle.right(4)
    counter +=1

turtle.exitonclick()