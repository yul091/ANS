import matplotlib.pyplot as plt

key = "ndcg@10"
read_txt = [
    "yelp-1000/mf-y-bpr/mf-bpr.txt",
    "yelp-1000/mf-y-fans/mf-fans-1.txt"
]

scores = []
for r in read_txt:
    with open(r, "r") as f:
        score = []
        for l in f:
            secs = l.split("|")
            for s in secs:
                if key + ":" in s:
                    score.append(float(s.split(":")[1]))
    scores.append(score)

for score in scores:
    plt.plot(score)
plt.show()