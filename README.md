# certainty-fm
This repository implements MC dropout certainty estimation for the Prithvi-100M foundation model.


## Setup

Please follow the setup instructions in this repository:
https://github.com/NASA-IMPACT/hls-foundation-os


Then, download the Sen1Floods11 dataset 

```
gsutil -m rsync -r gs://sen1floods11 .
```

There are some labeled examples provided in the test_labels directory.

To download more, you can use 

```
curl -O https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/{image_name}
```

## Usage
prithvi_mcdropout.py applies montecarlo dropout to the prithvi foundation model's inference on the sen1floods11 flood segmentation dataset. 

it writes results to a json file called metrics.json.

```
ARGS: 
--gpu (include this flag for inference with gpu.)
--stop (int) (include the flag with an integer n, to stop inference after image n)
--mc (int) (specify number of montecarlo dropout trials for certainty estimation. default is 3.)

e.g.
python prithvi_mcdropout.py --gpu --stop 2 --mc 2
```
