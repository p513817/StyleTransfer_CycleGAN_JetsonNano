# Pytorch深度學習框架X NVIDIA JetsonNano應用-cycleGAN風格轉換

![images](images/introduce.gif)

| 作者 | Chia-Chun, Chang |
| ---- | ---|
| 所屬單位  | Cavedu 教育團隊 |
| 開發日期  | 10909 |
| 基礎篇  | https://www.rs-online.com/designspark/jetson-nanogoogle-colabcyclegan-2-cn |
| 進階篇  | https://www.rs-online.com/designspark/jetson-nanogoogle-colabcyclegan-cn |

# 介紹
上次帶大家體驗了pix2pix並且將它應用在繪圖板上，畫出輪廓透過pix2pix去自動填上顏色，是不是很有趣呢？接下來我們再來玩一個風格轉換的經典作品Cycle GAN，並且將圖片轉換成梵谷風格圖片，那大家看到標題一定很好奇為什麼要叫做「偽」梵谷風格呢？因為看圖片就知道了，我總共只訓練了一百回合模型當然還沒訓練的很完整啦～不過這一百回合已經耗掉我一整個下午了！這次會教大家如何在Colab上實作、訓練，並且將自己訓練好的模型儲存到Jetson Nano上做應用。 




