from collections import OrderedDict
import my_modules as mm


# Accuracy------------------------------------------------------------------------------------------
def sklearn_metrics(actual,predicted,class_names=None,digits=4,print_result=False):
    print("Method: sklearn_metrics(actual,predicted,target_names=None,digits=4)")
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import precision_recall_fscore_support

    results = OrderedDict()

    if len(actual) != len(predicted):
        print("actual, predicted length not same: ",len(actual),len(predicted))

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

    if print_result:
        # print('accuracy_score: ','\x1b[1;31m',results["accuracy"],'\x1b[0m')
        print('accuracy_score: ',results["accuracy"])
        print("")
        print("\t\t\t Macro,\t\t\t Micro")
        print("\t\t\t -----,\t\t\t -----")
        # print("Precision:\t\t",results["precision_macro"],"\t",results["precision_micro"])
        # print("Recall:\t\t\t",results["recall_macro"],"\t",results["recall_micro"])
        print("f1:\t\t\t",results["f1_macro"],"\t",results["f1_micro"])
        # print("Precision: ",results["Precision"])
        # print("Recall: ",results["Recall"])
        # print("F1: ",results["F1"])
        print("")
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


def get_feature_result(train,n_classes,save_result=False):

    merged_result = OrderedDict()
    classes = mm.arrarr_bin([val["classes"] for id,val in train.items()],n_classes)

    knn_votes = mm.arrarr_bin([val["knn_votes"] for id,val in train.items()],n_classes,5)
    knn_votes_result = mm.sklearn_metrics(classes,knn_votes)
    if save_result:
        merged_result[str(save_result)+"_knn_votes"] = knn_votes_result
    else:
        merged_result["knn_votes"] = knn_votes_result
    correct_count = mm.classifier_agreement(classes,knn_votes)

    unique = mm.arrarr_bin([val["unique"]for id,val in train.items()],n_classes,1)
    unique_result = mm.sklearn_metrics(classes,unique)
    if save_result:
        merged_result[str(save_result)+"_unique"] = unique_result
    else:
        merged_result["unique"] = unique_result
    correct_count = mm.classifier_agreement(classes,unique)

    if save_result:
        mm.save_json(merged_result,str(save_result)+"feature",tag=True)

    return merged_result


def main():
    pass


if __name__ == "__main__": main()
