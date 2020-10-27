# File_System
File system simulation with elasticsearch.

**Requirements**
- Elasticsearch 
- Tensorflow
- dlib


Elasticsearch is used as CBIR. 

A trained Deep Convolutional Autoencoder is used to encode facial features which are obtained by cropping the face from the image. 

Key generation using encoded features. 

Search query using elasticsearch with threshold of 99%, and key to find the most relevant image which corresponds to the original user.

Compress functions to store the data by acquiring the id of the documents corresponding to the users.
