# CNV-FB
CNV-FB: A Feature bagging strategy based approach to detect copy number variants from NGS data

# Required Dependencies
## 1.python 3.8
    pandas 1.4.4
    numpy 1.23.1
    numba 0.56.2
    scikit-learn 1.1.2
    pysam 0.19.1
    rpy2 3.5.1
## 2.R 3.4.4
    DNAcopy
# Usage
## 1.Open the file `CNV-FB.py` and modify the variables `bamFilePath` and `refPath` inside;
```python
if __name__=="__main__":   
    bamFilePath = "./realData/NA12878.chrom21.SLX.maq.SRP000032.2009_07.bam"
    refPath='./realData/'
    n_estList=[30,40,50,60,70,80,90]
    run(bamFilePath, refPath, 70, '1',)
```
## 2.run the `CNV-FB.py`
# Real Datasets
The real datasets can be obtained in the following way.
- Clink this link:https://pan.baidu.com/s/10Qztg4QM-gK7xq8b35HaLQ?pwd=vsnh extraction code：vsnh

    