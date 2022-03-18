# pytorch-seq2seq-lstm(fixing
使用Pytorch實現Sequence to Sequence深度學習模型，本程式基於bentrevett/pytorch-seq2seq做修改，除了實作外，還有附上自己對該模型的理解。
## Recurrent Neural Network(RNN)
RNN最大的特性是會依照輸入資料的順序不同，而導致預測出來的結果不同，也就是RNN在學習時會考慮每個資料的出現順序，常被應用在輸入資料為序列的像是機器翻譯、文章分類與聊天機器人等文本相關應用。
下圖為其基本結構，其中<img src="http://chart.googleapis.com/chart?cht=tx&chl= a_{1}" style="border:none;">與<img src="http://chart.googleapis.com/chart?cht=tx&chl= a_{2}" style="border:none;">為存放資料的memory，在輸入資料時<img src="http://chart.googleapis.com/chart?cht=tx&chl= x_{1}" style="border:none;">與<img src="http://chart.googleapis.com/chart?cht=tx&chl= x_{2}" style="border:none;">經過隱藏層後資料會被存進memory<img src="http://chart.googleapis.com/chart?cht=tx&chl= a_{1}" style="border:none;">與<img src="http://chart.googleapis.com/chart?cht=tx&chl= a_{2}" style="border:none;">，再下個時間點輸入資料時，除了這個時間點的資料還會加上上個時間點memory裡面的值做運算，因此RNN會因為資料的輸入順序不同，而產生不同的輸出。
<!-- ![RNN](/image/RNN.png "RNN") -->
<img width="367" height="278" src="/image/RNN.png">
<a href="https://www.youtube.com/watch?v=xCGidAeyS4M&ab_channel=Hung-yiLee/">圖片來源: ML Lecture 21-1: Recurrent Neural Network</a>

## Long Short-Term Memory(LSTM)  
LSTM為RNN的變體，其與RNN相同會依照資料輸入順序的不同，而產生不同的預測的結果，特別的是LSTM新增了三個Gate:input gate, forget gate and output gate
## Sequence to Sequence
Sequence to Sequence由扮演encoder與decoder的兩個不同Long Short-Term Memory所組成
