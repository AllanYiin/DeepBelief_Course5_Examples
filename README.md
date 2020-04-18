

這個專案是基於我在tibame的深度學習課程(第五次改版)所設計的課程實作
<https://www.tibame.com/search?q=%E5%B0%B9%E7%9B%B8%E5%BF%97>

每個實作名稱前的編碼規劃是根據深度學習相關主題   
* epoch0 深度學習基礎概念與最佳化   
* epoch1 圖像識別   
* epoch2 目標檢測與追蹤   
* epoch3 語意分割/實例分割/全景分割   
* epoch4 非GAN的圖像生成、圖像修復、圖像翻譯   
* epoch5 生成式對抗網路(GAN)以及對抗式攻擊   
* epoch6 人臉相關應用   
* epoch7 基礎語言模型與語言預訓練模型   
* epoch8 Seq2Seq   
* epoch9 閱讀理解/智能問答   
* epochA 圖像與語言融合   
* epochB 強化學習  



最大的特色是由淺入深涵蓋深度學習主要在基本概論、機器視覺、自然語言以及強化學習等主題。  


為了讓不太熟程式撰寫的新手能夠樂於嘗試寫程式與讓已經有相當程度的熟手開發時更有效率，我覺得目前常用的keras這類的高階api仍距離我在工作時以及教學時的期望仍有一段距離，也因此我便著手設計了全新的api trident。 

 
trident的原意是三叉戟，象徵的我們之前一貫維持一個實作三種框架的平行代碼的風格(cntk, pytorch, tensorflow)，三種框架各自有不同的設計概念以及實現過程都有許多繁瑣的坑，而trident的目的就是希望能夠讓大家少掉進坑裡，盡量讓繁瑣的細節隱藏，讓訓練與推論pipline盡量抽象化進而可以變成容易套用的流程，目前trident主要是在pytorch上開發，在trident中可以使用類似keras的模式來設計網路結構，同時讓訓練流程變得簡潔卻強大，而且非常容易修改與維護，在這系列課程實作中，將都會以trident來實現。tensorflow版本的trident已經有部分實作支援，待pytorch部分開發告一段落時，我會盡速補上，從目前少部分支援tensorflow的實作，各位可以發現，基本上在trident中不管是哪種框架，幾乎95%以上代碼一致，而且代碼行數更少、更具可讀性。近期我也會將這部分的api代碼開源(因為現在都沒時間寫註解，所以還需要點時間)。

各位可以直接使用pip安裝trident

`
pip install tridentx   --upgrade  
`

引用trident的方法也很簡單，語法如下，需要透過環境變數來指定使用框架，目前支持pytorch 1.2以上以及tensorflow 2.0以上版本

`
import os  
os.environ['TRIDENT_BACKEND'] = 'pytorch'  
import trident as T  
from trident import *  
`'




之前第四次改版實作開源的政策是只有付費學員可以取得數據，在這次我們調整了政策，所有實作各位都可以透過trident api一行代碼就能夠取得。付費學員與開源使用者最大的差別會是在於tibame平台上會有每個實作的解說影片(每個實作解說影片約30-75分鐘長度) ，所以任何對深度學習有興趣的人都可以透過這個專案來學習深度學習。  



~~~~~~~~~~~~~~~~