# Setting up the image-retrieval engine

The engine depends on the following directories:
- The directory that contains the Flickr30k images (the images can be obtained at http://hockenmaier.cs.illinois.edu/DenotationGraph/)
- The directory that contains the annotations/sentences that describe the Flickr30k images (the zip file is included in the root directory of the project, it needs to be unzipped, after which the annotations.zip file inside needs to be unzipped as well) (Github repository: https://github.com/BryanPlummer/flickr30k_entities)
- The directory that contains the image features extracted from the Flickr30k images (the zip file is included in the root directory and needs to be unzipped)

The path to each of the three directories should be set in the ImageRetrievalEngine.py file.

The following command (**Mandatory**) allows the engine to encode the sentences and create 3 separate files containing the train/val/test image features:
```
python ImageRetrievalEngine.py -setup
```

**Optionally** the network can now be retrained entirely with the following command:
```
python ImageRetrievalEngine.py -train
```

Should you overwrite the training weights and later decide to revert back to the pretrained model weights, you can
copy-paste the model weights contained in *BestPerformingModel/* to the directory called *Models/*.


Finally, the following command (**Mandatory**) uses the trained network to create the embeddings for the test images, which are
then used by the engine when retrieving images based on captions:

```
python ImageRetrievalEngine.py -embed
```
# Usage

Once the image-retrieval engine has been set up, the following command starts up said engine:

```
python ImageRetrievalEngine.py -predict
```

The engine will prompt the user to enter a caption. The engine will then return the 10 most fitting images, and output the images
in *RetrievedImages/*. The user can then enter a new caption. The user can quit the program by entering:

```
:q
```

# Performance of the engine
The MAP@10 score is computed by entering the following command:

```
python ImageRetrievalEngine.py -map10
```
This computes the MAP@10 score on 100 test queries. The number of test queries can optionally also be specified as follows:

```
python ImageRetrievalEngine.py -map10 <nb_queries>
```

where <nb_queries> needs to be replaced by the desired number of queries.

# Notes

The **flickr30k_entities_utils.py** file in the root directory is copied over from the annotations/sentences directory (Github repository: https://github.com/BryanPlummer/flickr30k_entities).