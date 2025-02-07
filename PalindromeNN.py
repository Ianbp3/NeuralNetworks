import MyLibrary as ml

def wordinput(str):
    wordlen = len(str)
    if (wordlen > 8):
        print("Sorry, this word is too big")
        return []
    else:
        word = str.lower()
        wtoint = []

        for x in word:
            ch = (ord(x)-ord('0'))-48
            if (ch == -64):
                print("No Spaces Policy")
                return []
            if (ch > 26 or ch < 1):
                print("Not a letter")
                return []
            wtoint.append(ch)

        while wordlen != 8:
            wtoint.append(0)
            wordlen+=1
        return wtoint

#Programa
end = True
while end:
    word = input("Your word: ")
    if word == "null":
        end = False
    else:
        inputs = wordinput(word)
        print(inputs)
        if inputs != []:
            layer = ml.CapaDensa(8,10)
            layer.forward(inputs)
            relu = ml.ReLU()
            relu.forward(layer.salida)

            layer1 = ml.CapaDensa(10, 10)
            layer1.forward(relu.salida)
            relu1 = ml.ReLU()
            relu1.forward(layer1.salida)

            layer2 = ml.CapaDensa(10, 10)
            layer2.forward(relu1.salida)
            relu2 = ml.ReLU()
            relu2.forward(layer2.salida)

            layer3 = ml.CapaDensa(10, 10)
            layer3.forward(relu2.salida)
            relu3 = ml.ReLU()
            relu3.forward(layer3.salida)

            final_layer = ml.CapaDensa(10,1)
            final_layer.forward(relu3.salida)
            sigmoide = ml.Sigmoide()
            sigmoide.forward(final_layer.salida)
            print(sigmoide.salida)