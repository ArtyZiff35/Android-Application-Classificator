from matplotlib import pyplot as plt

# x axis values
x = [2,3,4,5,6,7,8,9,10]
# corresponding y axis values
y_avg = [0.8588, 0.7560, 0.6948, 0.6290, 0.6071, 0.5631, 0.5493, 0.5355, 0.5248]
y_max = [0.9322, 0.8271, 0.7943, 0.6805, 0.6609, 0.6225, 0.6340, 0.5992, 0.6111]
# plotting the points
plt.plot(x, y_avg)

# naming the x axis
plt.xlabel('Number of Categories')
# naming the y axis
plt.ylabel('Average accuracy')

# giving a title to my graph
plt.title('Average model accuracy according to # of categories')

# function to show the plot
plt.show()