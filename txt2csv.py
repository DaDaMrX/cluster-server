with open('chat-short-20w.txt') as f:
    data = f.read().strip().split('\n\n')

for i in range(len(data)):
    data[i] = data[i].strip().split('\n')
    for j in range(len(data[i])):
        data[i][j] = [str(i), data[i][j][0], data[i][j][2:]]
        data[i][j] = ','.join(data[i][j])

with open('chat.csv', 'w') as f:
    for conv in data:
        for sent in conv:
            f.write(sent + '\n')

print(len(data))