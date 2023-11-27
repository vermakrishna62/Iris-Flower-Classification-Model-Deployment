
import joblib as jb

model = jb.load('./savedModel/iris_model.joblib')

def iris_classification(request):
    
    if request.method == 'POST':
        
        data = request.POST
        
        sepal_lt = data.get('sepalLength')
        sepal_wt = data.get('sepalWidth')
        petal_lt = data.get('petalLength')
        petal_wt = data.get('petalWidth')
        
        feature_values = {
                "sepal_length": sepal_lt,
                "sepal_width": sepal_wt,
                "petal_length": petal_lt,
                "petal_width": petal_wt
            }
            
        # Make predictions using the model
        y_pred = model.predict([list(feature_values.values())])[0]
                    
        if y_pred == 0:
            y_pred = 'Setosa'
        elif y_pred == 1:
            y_pred = 'Versicolor'
        elif y_pred == 2:
            y_pred = 'Virginica'
        
        return render(request,'index.html',{'op':y_pred})
    
    
    return render(request,'index.html')
