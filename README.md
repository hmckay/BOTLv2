# BOTL
This git repo has data, data generators and code for the Bi-directional Online Transfer Learning framework (BOTL). This is an updated version of the BOTL framework (see original in https://github.com/hmckay/BOTL).
BOTL was first introduced in [[1]](#1), and extended in [[2]](#2).

New functionality includes:
  - using Principal Angles between subspaces to estimate the conceptual similarity between base models
  - parameterised conceptual similarity thresholding for base model selection
  - parameterless conceptual clustering for base model selection using Self-Tuning Spectral Clustering (see https://github.com/wOOL/STSC)

## Running BOTL
To use BOTL, run via `controller.py`
- input parameters can be found using `python3 controller.py --help`, this specifies the underlying concept drift detector used, the method of selecting/culling base models, input dataset type, etc.


## Available data and data generators and BOTL implementations:
  - Following distance data for 6 journeys (2 drivers).
  - Drifting hyperplane data generator
  - Smart home heating simulation (with real world weather data)

Note the underlying framework is the same for all three implementations. For ease of reproducibility all three versions have been added.

## AWPro
AWPro is a concept drift detection algorithm that combines aspects of ADWIN and RePro that better suit the BOTL framework. AWPro was first introduced in [[2]](#2).

## Parameter Analysis
Parameter analysis to consider the impact of the parameter values of underlying concept drift detection strategies, and how they impact the BOTL framwork is discussed in [[2]](#2) (and available in: https://github.com/hmckay/BOTL, see `parameterAnalysis.pdf`).


# File structure
The BOTL framework is available for Hyperplane, Heating and FollowingDistance datasets (see `reproducibility.pdf` in https://github.com/hmckay/BOTL for more information of these datasets).
BOTL has been implemented using three underlying concept drift detection algorithms: RePro, ADWIN and our own drift detector, AWPro. 

RePro implementations are situated in:
`sourceRePro.py`
ADWIN implementations are in:
`sourceADWIN.py`
AWPro implementations are in:
`sourceAWPro.py`


Code structure:
* *`controller.py`*: manages the creation of domains in the framework and is used to transfer models between domains (sources).
* *`source.py`*: a source domain. Uses one of the underlying concept drift detection strategies to detect concept drifts that occur locally to a domain.
* *`Models/createModel.py`*: used to create local models and make predictions without knowledge transfer. Predictions without knowledge transfer are used to detect concept drifts
* *`Models/modelMultiConceptTransfer.py`*: used to make overarching predictions by combining the locally learnt model with models transferred from other domains (more detail below)
* *`Models/modelMultiConceptTransferHistory.py`*: used by BOTL to keep track of source models and to identify when a model is considered stable (therefore can be transferred to other domains). Also used by BOTL implementation with RePro and AWPro as underlying drift detectors to keep track of historical models and concept transitions. Allows previously learnt models to be prioritised over creating new models

# BOTL with Culling (BOTL-C)
Three BOTL-C variants are included in this repository: BOTL with PThresh (also referred to BOTL-C.I in previous repository, model culling based on performance), BOTL with MIThresh (also referred to as BOTL-C.II in the previous repository, model culling based on performance and diversity), and BOTL with PAThresh (model culling using Principal Angles between subspaces to estimate conceptual similarity) . Each of these are implemented in `Models/modelMultiConceptTransfer.py`. BOTL with PThresh is used when the parameter `weightType = 'OLSFE'`, BOTL with MIThresh is used when the parameter `weightType = 'OLSFEMI'`, and BOTL with PAThresh is used when the parameter `weightType = OLSFEPA`. In order to use these implementations, additional parameters are needed, which are set in `controller.py` as follows:
- *Performance threshold*: this was the original culling parameter, and therefore is denoted by parameter `CThresh` in `controller.py`
- *Mutual Information threshold*: this parameter is denoted by `MIThresh` in `controller.py`
- *Conceptual Similarity threshold*: this parameter is denoted by `PAThresh` in `controller.py`


# Source Code
The BOTL framework has been created using various code from other sources. ADWIN and AWPro implementations (which uses ADWIN as a basis for drift detection) are based upon the implementation available: https://github.com/rsdevigo/pyAdwin. This code is included in `datasetBOTL/BiDirTansfer/pyadwin/`

Self-Tuning Spectral Clustering has been created based on the implementation available: https://github.com/wOOL/STSC.

## References
<a id="1">[1]</a> 
McKay, H., Griffiths, N., Taylor, P., Damoulas, T. and Xu, Z., 2019. Online Transfer Learning for Concept Drifting Data Streams. In BigMine@ KDD.

<a id="2">[2]</a>
McKay, H., Griffiths, N., Taylor, P., Damoulas, T. and Xu, Z., 2020. Bi-directional online transfer learning: a framework. Annals of Telecommunications, 75(9), pp.523-547.

