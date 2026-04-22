import time
print("AI mind Rider")
time.sleep(1)

print("Thinking of a number between 1 to 100")
input("press Enter when you're ready...")

low = 1
high = 100
while low<= high:
    mid = (low+high) // 2   # finding mid value
    answer = input(f"\nIs your number greater than {mid}?(y/n): ")
    
    if answer.lower() == "y":
        low = mid + 1
    else:
        high = mid - 1

print("\nAI Predection complete")
print("Your number is : ", low) 


# this code typically searching for a binary search...
# And it find a mid value for finding number and it working procudure 7 step....