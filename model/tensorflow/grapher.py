import matplotlib.pyplot as plt
import re


xs =[]
trainingYs = []
valYs = []

with open('decay_logs/resnet_augment7.txt', 'r') as f:
    print f
    content = f.readlines()
    print content
    content = [x.strip() for x in content] 
    print content
    odd = 1
    for line in content:
        nums = []
        items = line.split()
        if len(items) > 10:
            if odd == 1:
                xs.append( int(re.findall(r'\d+(?:\.\d+)?', items[1])[0]))
                trainingYs.append( 1 - float(items[11]))
                odd = 2
            else:
                valYs.append(1 - float(items[11]))
                odd = 1

print len(xs)
print len(valYs)
print len(trainingYs[:45])
xs = xs[:45]
fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title('Top-5 Error Throughout Training')
ax.set_xlabel("Iteration")
ax.set_ylabel("Error")
line1, = plt.plot(xs, trainingYs[:45], 'b', label = "Training Error")

line2, = plt.plot(xs, valYs, 'r', label = "Validation Error")

plt.legend([line1, line2], ["Training Error", "Validation Error"], prop={'size': 20})
plt.show()

        # for i in line.split():
        #     i = re.findall(r'\d+(?:\.\d+)?', i)
        #     print i
        #     if len(i)==1:
        #         nums.append(int(i[0]))
# with open('submission.txt', 'w') as dest:
#   with open('temp.txt', 'r') as filename:
#       for image in filename:
#           classification = sess.run(y,feed_dict={x: image})
#           prediction=tf.argmax(y,1)
#           prediction.eval(feed_dict={x: image}, sess=sess)
#           classifications = tf.nn.top_k(y, k=5)
#           top5strings = ''
#           for item in classifications.indices[0]:
#               top5strings+=str(item)+' '
#           dest.write('%s%s\n' % (image.rstrip('\n'), top5strings))
