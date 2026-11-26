from processor import load_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score




def tune_knn(x_train, y_train, x_val, y_val):
    k_values = [1,3,5,7,9,11,13,15,17,19,21,23,25]
    best_k = None
    best_val_acc = 0.0
    best_knn_model = None

    for k in k_values:

        knn_model = KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
        knn_model.fit(x_train, y_train)

        train_pred = knn_model.predict(x_train)
        val_pred = knn_model.predict(x_val)
        
        train_acc = accuracy_score(y_train,train_pred)
        val_acc = accuracy_score(y_val,val_pred)
        print(f"{k:<10} | {train_acc:<12.4f} | {val_acc:<12.4f}")

        if val_acc >best_val_acc :

            best_val_acc = val_acc
            best_k = k
            best_knn_model =knn_model
        
    print("-" * 40)
    print(f"Best K: {best_k} | Best Validation Accuracy: {best_val_acc:.4f}")
    return best_knn_model, best_k, best_val_acc

if  __name__ == "__main__":
    x_train, x_val, x_test, y_train, y_val, y_test, le, scaler = load_data()
    best_knn_model, best_k, best_val_acc = tune_knn(x_train, y_train, x_val, y_val)
    

