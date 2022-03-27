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

其中<img src="http://chart.googleapis.com/chart?cht=tx&chl= z_i" style="border:none;">為輸入激活函數的值，<img src="http://chart.googleapis.com/chart?cht=tx&chl= a_i" style="border:none;">為<img src="http://chart.googleapis.com/chart?cht=tx&chl= z_i" style="border:none;">經過激活函數輸出的值，<img src="http://chart.googleapis.com/chart?cht=tx&chl= f'(z_i)" style="border:none;">為激活函式對<img src="http://chart.googleapis.com/chart?cht=tx&chl= z_i" style="border:none;">的偏微分。可以看出，在輸入序列很長的情況下，若要得到<img src="http://chart.googleapis.com/chart?cht=tx&chl= z_1" style="border:none;">的梯度來更新權重W，其梯度會是多項連乘，這時若W大於1，梯度會非常大，稱為梯度爆炸，反之若W小於1，梯度會接近0造成網路根本沒法更新，稱為梯度消失。

## Long Short-Term Memory(LSTM)  
LSTM為RNN的變體，其目的就是改善RNN的梯度消失與梯度爆炸的問題，與RNN相同，LSTM會依照資料輸入順序的不同，而產生不同的預測的結果，特別的是，LSTM新增了三個Gate:input gate, forget gate and output gate，下圖為LSTM的範例:

<img width="419" height="302" src="/image/LSTM.png">

<a href="https://www.youtube.com/watch?v=xCGidAeyS4M&ab_channel=Hung-yiLee/">圖片來源: ML Lecture 21-1: Recurrent Neural Network</a>

input gate負責控制輸入資料要進入多少到memory裡，forget gate負責控制上個時間點的memory要保留多少，output gate則負責控制要從memory輸出多少資訊，而這些gate的控制全由類神經網路的權重來決定，因此LSTM可以學習一個資訊要存在memory裡多久、在何時要記憶與輸出新的資訊，以此來避免梯度爆炸與梯度消失的問題。

LSTM同樣存在一些問題，將LSTM應用在實際問題時，其輸出的序列不能大於輸入序列，像是輸入一段文章產生該文章的分類或是輸入一個句子輸出每個詞的詞性這些應用都沒有限制，但像是翻譯或問答等問題，其輸出序列可能會比輸入序列還要長，這時候LSTM就無法直接應用。

## Sequence to Sequence
為了解決LSTM其輸出序列長度不能大於輸入序列的問題，Sequence to Sequence誕生了，Sequence to Sequence由扮演encoder與decoder的兩個不同LSTM所組成，encoder其輸入為序列，輸出為向量，
decoder輸入為向量，輸出為序列，

## To Do List
### Attention-Based Sequence to Sequence
### self-attention(Attention is all you need)
### Reinforcement Learning
