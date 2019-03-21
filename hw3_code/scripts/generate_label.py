w = open("list/NULL_val_label", 'w')
with open("../all_val.lst", 'r') as f:
    for line in f:
        x = line.strip().split(' ')[1]
        x = 1 if x=="NULL" else 0
        w.write("{}\n".format(x))
print("Done")