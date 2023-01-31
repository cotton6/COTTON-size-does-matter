
FILE=result/Top_1024x768_COTTON/weights
if ! test -d "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-2RuuPLfrJ_Uia_BKrk2Pj58BSdWofgc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-2RuuPLfrJ_Uia_BKrk2Pj58BSdWofgc" -O result/Top_1024x768_COTTON/weights.zip && rm -rf /tmp/cookies.txt
    unzip result/Top_1024x768_COTTON/weights.zip -d result/Top_1024x768_COTTON/
fi

FILE=result/Top_1024x768_DressCode/weights
if ! test -d "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eLSNbixRyVSa_EmZgCJ0niOS5V66F2U1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eLSNbixRyVSa_EmZgCJ0niOS5V66F2U1" -O result/Top_1024x768_DressCode/weights.zip && rm -rf /tmp/cookies.txt
    unzip result/Top_1024x768_DressCode/weights.zip -d result/Top_1024x768_DressCode/
fi

