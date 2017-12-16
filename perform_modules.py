from collections import OrderedDict


# Accuracy------------------------------------------------------------------------------------------
def sklearn_metrics(actual,predicted,class_names=None,digits=4):
    print("Method: sklearn_metrics(actual,predicted,target_names=None,digits=4)")
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import precision_recall_fscore_support

    results = OrderedDict()

    results["f1_macro"] = f1_score(actual,predicted,average='macro')
    results["f1_micro"] = f1_score(actual,predicted,average='micro')
    results["accuracy"] = accuracy_score(actual,predicted)
    results["precision_macro"] = precision_score(actual,predicted,average='macro')
    results["precision_micro"] = precision_score(actual,predicted,average='micro')
    results["recall_macro"] = recall_score(actual,predicted,average='macro')
    results["recall_micro"] = recall_score(actual,predicted,average='micro')
    results["F1_class"] = precision_recall_fscore_support(actual,predicted)[2].tolist()
    results["Precision_class"] = precision_recall_fscore_support(actual,predicted)[0].tolist()
    results["Recall_class"] = precision_recall_fscore_support(actual,predicted)[1].tolist()

    # print('accuracy_score: ','\x1b[1;31m',results["accuracy"],'\x1b[0m')
    print('accuracy_score: ',results["accuracy"])
    print("\t\t\t Macro,\t\t\t Micro")
    print("\t\t\t -----,\t\t\t -----")
    # print("Precision:\t\t",results["precision_macro"],"\t",results["precision_micro"])
    # print("Recall:\t\t\t",results["recall_macro"],"\t",results["recall_micro"])
    print("f1:\t\t\t",results["f1_macro"],"\t",results["f1_micro"])
    # print("Precision: ",results["Precision"])
    # print("Recall: ",results["Recall"])
    # print("F1: ",results["F1"])
    if class_names:
        print(classification_report(y_true=actual,y_pred=predicted,target_names=class_names,digits=digits))
    else:
        print(classification_report(y_true=actual,y_pred=predicted,digits=digits))

    print("\n")
    return results


def accuracy_multi(actual,predicted,n_classes,multi=True):
    """Calculates (Macro,Micro) precision,recall"""
    print("Method: accuracy_multi(actual,predicted,multi=True)")
    if len(actual) != len(predicted):
        print("** length does not match: ",len(actual),len(predicted))
    class_count=[0] * n_classes
    for i in range(len(actual)):
        if multi:
            for pred_label in predicted[i]:
                if pred_label in actual[i]:
                    class_count[pred_label]=class_count[pred_label]+1
        else:
            if actual[i] == predicted[i]:
                class_count[predicted[i]]=class_count[predicted[i]]+1
    print("Predicted counts per class:\t",class_count)


def main():
    pass


if __name__ == "__main__": main()
