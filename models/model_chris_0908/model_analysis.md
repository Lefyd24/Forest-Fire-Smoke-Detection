Based on the graphs provided during the training of the model the following conclusions can be drawn regarding the performance of the model per metric:
# 1. Precision - Recall Curve
The precision-recall curve shows the tradeoff between precision and recall for different threshold. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate. High scores for both show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall).<br>

#### Graph 1.1: Precision-Recall Curve
<img src="PR_curve.png" width="1000" height="500" /><br>

`In our case:`
> * *`High Precision at Low Recall`*: At the beginning of the curve (towards the left), the precision is very high, suggesting that the model is highly confident in its positive predictions for a limited set of instances. However, as we try to capture more positive instances (increasing recall), the precision starts to drop, indicating that the model begins to make more false positive errors. <br>
> * *`Curve Shape`*: The curve is relatively smooth but has some dips, suggesting that there are certain thresholds where the precision drops more significantly. This could be due to various reasons like class imbalance or certain classes being harder to predict than others.
> * *`Area Under the Curve (AUC)`*: A larger area under the PR curve indicates better performance. From the graph 1.1 above, the curve seems to cover a good portion of the graph, indicating decent performance.<br>
> * *`Curve's Decline`*: Towards the extreme right of the graph, there's a steeper decline. This indicates that as we try to achieve <b>very high recall, the precision drops off more rapidly<b>. This could be due to the model struggling with more challenging or ambiguous instances as said in bibliography. In our case, since smoke and fire are quite ambiguous, in both shape and color, it is expected that the model will struggle to predict them with high confidence.</b><br>