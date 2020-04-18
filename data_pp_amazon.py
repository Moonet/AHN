import gzip
import random
import re
from collections import defaultdict
from os import listdir
from os.path import isfile, join
import pickle

bid2tid = defaultdict(list)
uid2tid = defaultdict(list)
u2i = set()
t2t = {}


path = '/amazon_dataset/amazon_subset/'
cata_files = ['reviews_Automotive_5.json.gz']
sentences = []
labels = []
rid = 0

print(cata_files)
for i in range(len(cata_files)):
    catagory = cata_files[i]
    new_path = path + catagory
    g = gzip.open(new_path, 'r')
    for l in g:
        review = eval(l)
        text = review['reviewText']
        text = re.sub(r'http\S+', '', text)
        text = text.strip().rstrip().lower()
        if text != '':
            rating = review['overall']
            uid = review['reviewerID']
            bid = review['asin']
            date = review['unixReviewTime']
            u2i.add((uid, bid, (date, rid), rating))
            t2t[rid] = text
            rid += 1


print('Extract information Done')
# split to training, validation, testing

###############
edges = {}
points = {}


def addEdge(p1, p2):
    if p1 in edges.keys():
        edges[p1].append(p2)
    else:
        edges[p1] = [p2]
    if p2 in edges.keys():
        edges[p2].append(p1)
    else:
        edges[p2] = [p1]
    if p1 in points.keys():
        points[p1] += 1
    else:
        points[p1] = 1
    if p2 in points.keys():
        points[p2] += 1
    else:
        points[p2] = 1


def aveDegree():
    n = len(points)
    print(n)
    s = sum(points.values())
    return float(s) / float(n)


def prune(kcore):
    for k in list(points.keys()):
        v = points[k]
        if v < kcore:
            for p in edges[k]:
                if p in points.keys():
                    points[p] -= 1
            del points[k]


def evolve(kcore):
    n = len(points)
    prune(kcore)
    iter = 0
    while len(points) != n:
        n = len(points)
        prune(kcore)
        iter += 1
    print(iter)


for link in u2i:
    addEdge(link[0], link[1])

evolve(5)

aveDegree()
for item, n in points.items():
    if n < 5:
        print(item)

print("[SUCCESS] Get all users who has at least 5 reviews!")

###############
u2i = list(u2i)
total_links = len(u2i)
train = u2i[ : int(0.8*total_links)]
non_train = u2i[int(0.8*total_links):]


new_train = []
for item in train:
    uid = item[0]
    bid = item[1]
    date, rid = item[2]
    if uid in points.keys() and bid in points.keys():
        new_train.append(item)
        uid2tid[uid].append((date, rid))
        bid2tid[bid].append((date, rid))

print(len(new_train))

u_num = 0
b_num = 0
new_non_train = []
for item in non_train:
    uid = item[0]
    bid = item[1]
    if uid in uid2tid.keys() and bid in bid2tid.keys():
        new_non_train.append(item)

print(len(new_non_train))

print("first occuring user number: " + str(u_num))
print("first occuring item number: " + str(b_num))


for bid, btid in bid2tid.items():
    btid = sorted(btid, key=lambda x: x[0])
    bid2tid[bid] = btid

for uid, utid in uid2tid.items():
    utid = sorted(utid, key=lambda x:x[0])
    uid2tid[uid] = utid

avail_uids = uid2tid.keys()
avail_bids = bid2tid.keys()
print("number of available users: {}".format(len(avail_uids)))
print("number of available items: {}".format(len(avail_bids)))


print("number of user to item: {}".format(len(new_train)))

length = 0
for item in u2i:
    rid = item[2][1]
    length += len(t2t[rid].split())

print("avg of words per review: {}".format(length/len(u2i)))
#
with open('uid2tid.pkl', 'wb') as f:
    pickle.dump(uid2tid, f)

with open('bid2tid.pkl', 'wb') as f:
    pickle.dump(bid2tid, f)

print('Done')

dev = new_non_train[: int(0.5*len(new_non_train))]
test = new_non_train[int(0.5*len(new_non_train)):]
#
#


new_t2t = {}
rid_list = set()
for item in new_train:
    rid_list.add(item[2][1])

for id, txt in t2t.items():
    if id in rid_list:
        new_t2t[id] = txt

print(len(new_t2t))
with open('train.pkl', 'wb') as f:
    pickle.dump(new_train, f)
with open('dev.pkl', 'wb') as f:
    pickle.dump(dev, f)
with open('test.pkl', 'wb') as f:
    pickle.dump(test, f)
print('Done')

with open('t2t.pkl', 'wb') as f:
    pickle.dump(new_t2t, f)

print('Done')
