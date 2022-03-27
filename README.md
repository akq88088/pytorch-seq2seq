# pytorch-seq2seq-lstm(施工中
使用Pytorch實現Sequence to Sequence深度學習模型，本程式基於bentrevett/pytorch-seq2seq做修改，除了實作外，還有附上自己對該模型的理解。
## Recurrent Neural Network(RNN)
RNN最大的特性是會依照輸入資料的順序不同，而導致預測出來的結果不同，也就是RNN在學習時會考慮每個資料的出現順序，常被應用在輸入資料為序列的像是機器翻譯、文章分類與聊天機器人等文本相關應用。
下圖為一個RNN的簡單範例，其中<img src="http://chart.googleapis.com/chart?cht=tx&chl= a_{1}" style="border:none;">與<img src="http://chart.googleapis.com/chart?cht=tx&chl= a_{2}" style="border:none;">為存放資料的memory，在進行前向傳播時輸入資料<img src="http://chart.googleapis.com/chart?cht=tx&chl= x_{1}" style="border:none;">與<img src="http://chart.googleapis.com/chart?cht=tx&chl= x_{2}" style="border:none;">經過隱藏層的激活函數後，會被累加至<img src="http://chart.googleapis.com/chart?cht=tx&chl= a_{1}" style="border:none;">與<img src="http://chart.googleapis.com/chart?cht=tx&chl= a_{2}" style="border:none;">這兩個memory裡面，在下個時間點輸入的資料，除了目前的輸入資料還會再加上儲存在<img src="http://chart.googleapis.com/chart?cht=tx&chl= a_{1}" style="border:none;">與<img src="http://chart.googleapis.com/chart?cht=tx&chl= a_{2}" style="border:none;">這兩個memory裡面的值，因此RNN會因為資料的輸入順序不同，而產生不同的輸出。
<img width="367" height="278" src="/image/RNN.png">

<a href="https://www.youtube.com/watch?v=xCGidAeyS4M&ab_channel=Hung-yiLee/">圖片來源: ML Lecture 21-1: Recurrent Neural Network</a>

RNN存在梯度消失與梯度爆炸的問題，這邊我們舉一個例子，下圖為一個RNN範例，輸入長度為1000且所有值都為1的序列，RNN會產生
<img src="http://chart.googleapis.com/chart?cht=tx&chl= y_{1}" style="border:none;">到<img src="http://chart.googleapis.com/chart?cht=tx&chl= y_{1000}" style="border:none;">共1000筆輸出。

<img width="404" height="286" src="/image/rnn_problem_example.png">

這邊我們只關注連到memory的權重W，當RNN要進行倒傳遞(backpropagation through time)來修正權重W時，其梯度簡單表示如下式:

<img src="http://chart.googleapis.com/chart?cht=tx&chl= \frac{\partial{a_{999}}}{\partial{z_1}}=f'(z_{999})wf'(z_{998})w...f'(z_{1})" style="border:none;">

其中<img src="http://chart.googleapis.com/chart?cht=tx&chl= z_i" style="border:none;">為輸入激活函數的值，<img src="http://chart.googleapis.com/chart?cht=tx&chl= a_i" style="border:none;">為經過激活函數輸出的值，<img src="http://chart.googleapis.com/chart?cht=tx&chl= f'(z_i)" style="border:none;">為激活函式對<img src="http://chart.googleapis.com/chart?cht=tx&chl= z_i" style="border:none;">的偏微分。可以看出，在輸入序列很長的情況下，若要更新權重W，其梯度會是多項連乘，這時若W大於1，其梯度會非常大，稱為梯度爆炸，反之若梯度小於1，梯度會接近0造成網路根本沒法更新，稱為梯度消失。

## Long Short-Term Memory(LSTM)  
LSTM為RNN的變體，其與RNN相同會依照資料輸入順序的不同，而產生不同的預測的結果，特別的是LSTM新增了三個Gate:input gate, forget gate and output gate

## Sequence to Sequence
Sequence to Sequence由扮演encoder與decoder的兩個不同Long Short-Term Memory所組成

## To Do List
### Attention-Based Sequence to Sequence
### self-attention(Attention is all you need)
### Reinforcement Learning
