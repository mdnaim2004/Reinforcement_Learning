class car:
    name = ""
    color = ""

    def start():
        print("Starting the car, and turn on the engine")

car.name = "BMW"
car.color = "Dark.nevy-Blue"

print("Name of the color : ", car.name)
print("The car colors : ", car.color)

car.start()
print(dir(car))