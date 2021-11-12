# Generate Sememe Representation via Building a Large Semantic Graph

python==3.6.5
mecab-python3==1.0.4
scikit-learn==1.0.1
scipy==1.7.1
nltk==3.6.5
gensim==4.1.2
biterm==0.1.5

cpp gsl==2.4.0

* The GSL package is used and can be downloaded at http://www.gnu.org/software/gsl/

## Preprocessing

- cd ./preprocessing
- python ./src/download-wiki-extract.py
- bash ./src/wiki-preprocessing.sh
- python ./local_processing.py
- python ./entity_processing.py 
- python ./tokenizer.py

## Building a large semantic graph

- cd ./graph_building
- python ./LSA.py
- python ./building.py
- cat word.graph doc.graph topic.graph > sememe.graph

## Graph embedding

- cd ./grapg_embedding
- ./line -train sememe.graph -output ./sememe.embedding -binary 0 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20
