檔案架構:
  - run.sh: 會執行一連串指令作為參數實驗
  - main.py: 處理args的argument，並呼叫src.HDR來執行主要的運算
  - src:
    - HDR.py: 負責alignment及實作Debevec's method。
    - tonemap.py: 負責將.hdr轉成ldr
  - images
    - raw: 需要進行處理的影像都放置在此資料夾


執行方式:
由於主程式為main.py因此以下介紹其使用上的argument
僅有開頭沒有"--"的argument為必須輸入，其他皆設有預設值方便使用者使用
  input_dirpath		讀取原始不同曝光時間影像的資料夾
  output_dirpath	輸出hdr影像、tonemap影像、radiance map影像的資料夾(但實際輸出影像的資料夾會自己根據argument來進行命名)
  --lamda		Debevec's method所使用的lambda，預設為15
  --align		要不要先將影像進行alignment
  --sampling		取樣的步伐大小,預設為21
  --alignTime		若要將影像進行alignment，要遞迴幾次，預設為5

ex. 輸入資料夾為images/raw/img2 輸出資料夾為result lambda為10 要使用alignment sampling stride為21 alignTime 為5
python main.py images/raw/img2 result --lamda 10 --align --sampling 21 --alignTime 5