
FILE=CIHP_PARSING/checkpoint
if ! test -d "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AmS_ybeyIPAJ_wB2trB4HoxeEclGaYyQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AmS_ybeyIPAJ_wB2trB4HoxeEclGaYyQ" -O CIHP_PARSING/checkpoint.zip && rm -rf /tmp/cookies.txt
    unzip CIHP_PARSING/checkpoint.zip -d CIHP_PARSING
fi

FILE=U2Net/saved_models
if ! test -d "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1IGkF9yQ2wb-I_iBqXTDUr4qLkZWJW4me' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1IGkF9yQ2wb-I_iBqXTDUr4qLkZWJW4me" -O U2Net/saved_models.zip && rm -rf /tmp/cookies.txt
    unzip U2Net/saved_models.zip -d U2Net
fi

FILE=Sleeve_Classifier/weights
if ! test -d "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yqGVPP896VcU-hOT2JtoQ_C_S1LQAt20' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yqGVPP896VcU-hOT2JtoQ_C_S1LQAt20" -O Sleeve_Classifier/weights.zip && rm -rf /tmp/cookies.txt
    unzip Sleeve_Classifier/weights.zip -d Sleeve_Classifier
fi

FILE=lower_clf/weights
if ! test -d "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MG3kKV0-3kTO32IDBbCMVqwW0EVlTGWh' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1MG3kKV0-3kTO32IDBbCMVqwW0EVlTGWh" -O lower_clf/weights.zip && rm -rf /tmp/cookies.txt
    unzip lower_clf/weights.zip -d lower_clf
fi

FILE=Cloth2Skeleton/weights
if ! test -d "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=15TjQbsWekmDZgjc6jbnx6zfwlzqJNL5Q' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=15TjQbsWekmDZgjc6jbnx6zfwlzqJNL5Q" -O Cloth2Skeleton/weights.zip && rm -rf /tmp/cookies.txt
    unzip Cloth2Skeleton/weights.zip -d Cloth2Skeleton
fi

FILE=ClothSegmentation/weights
if ! test -d "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kiAWsM6WmcoNyp864dqrzl1n4HfEe7bH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kiAWsM6WmcoNyp864dqrzl1n4HfEe7bH" -O ClothSegmentation/weights.zip && rm -rf /tmp/cookies.txt
    unzip ClothSegmentation/weights.zip -d ClothSegmentation
fi

FILE=Self-Correction-Human-Parsing/exp-schp-201908301523-atr.pth
if ! test -f "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13iCVTqAATnBbEVMIKRIwzLkV4czMt0kd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13iCVTqAATnBbEVMIKRIwzLkV4czMt0kd" -O Self-Correction-Human-Parsing/exp-schp-201908301523-atr.pth && rm -rf /tmp/cookies.txt
fi

