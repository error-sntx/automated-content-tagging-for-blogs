from pathlib import Path
from collections import Counter
import csv,json,re,html,math
BASE=Path(__file__).resolve().parent
def load(name):
 with (BASE/'data'/name).open(newline='') as f:return list(csv.DictReader(f))
def tokens(text):
 text=re.sub(r'<[^>]+>',' ',text)
 return re.findall(r'[a-z]+',html.unescape(text).lower())
articles=load('articles.csv');labels={r['article_id']:r for r in load('labels.csv')};taxonomy=load('taxonomy.csv')
counts={r['category_id']:Counter() for r in taxonomy};totals=Counter();vocab=set()
for a in articles:
 if a['split']!='train':continue
 words=tokens(a['title']+' '+a['content']);vocab.update(words)
 for category in labels[a['article_id']]['category_ids'].split('|'):counts[category].update(words);totals[category]+=len(words)
rows=[];scores={}
for a in articles:
 words=Counter(tokens(a['title']+' '+a['content']))
 likelihood={k:sum(n*math.log((counts[k][w]+1)/(totals[k]+len(vocab))) for w,n in words.items()) for k in counts}
 best=max(likelihood,key=likelihood.get)
 # Single-label starting point; students extend to multi-label prediction.
 tags=next(r['canonical_tags'] for r in taxonomy if r['category_id']==best)
 rows.append(dict(article_id=a['article_id'],category_ids=best,tags=tags))
for split in ['validation','test']:
 ids={a['article_id'] for a in articles if a['split']==split};tp=fp=fn=0;exact=0
 for r in rows:
  if r['article_id'] not in ids:continue
  p=set(r['category_ids'].split('|'));y=set(labels[r['article_id']]['category_ids'].split('|'))
  tp+=len(p&y);fp+=len(p-y);fn+=len(y-p);exact+=p==y
 scores[split]={'micro_f1':2*tp/(2*tp+fp+fn),'exact_match':exact/len(ids),'articles':len(ids)}
with (BASE/'predictions.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(BASE/'metrics.json').write_text(json.dumps(scores,indent=2))
print('Created predictions.csv and metrics.json')
