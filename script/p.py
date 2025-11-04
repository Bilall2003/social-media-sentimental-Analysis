text="""my name is bilal
and my class is 5.
"""
print(text)
word1=text.split()
print(word1)

tex=" ".join(word1)
print(tex)

word2=tex.split()
print(word2)


from collections import Counter

count=Counter(word2)
print(count.most_common(10))