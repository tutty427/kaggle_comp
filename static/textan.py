from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# querypath = "D:/workspace_py/pyproject/static/dataset3"
#
# q_file = open(querypath, 'r')
# txt_content = q_file.readline()
# q_file.close()

txt_content = ["I have a apple","I have a pen","I have a apple pen"]
model = CountVectorizer()
result = model.fit_transform(txt_content)
result2 = model.vocabulary_.get(u'apple')

print(result.shape)
print(result2)