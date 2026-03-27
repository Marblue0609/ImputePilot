from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('hello/', views.hello, name='hello'),
    
    # Dashboard
    path('dashboard/status/', views.get_model_status, name='model_status'),
    path('dashboard/benchmarks/', views.get_benchmarks, name='benchmarks'),
    
    # Pipeline
    path('pipeline/upload/', views.upload_training_data, name='upload_training'),
    path('pipeline/clustering/', views.run_clustering, name='clustering'),
    path('pipeline/labeling/', views.run_labeling, name='labeling'),
    path('pipeline/features/', views.run_features, name='features'),
    path('run-features-async/', views.run_features_async, name='features_async'),
    path('pipeline/modelrace/', views.run_modelrace, name='modelrace'),
    
    # Baseline Training
    path('baseline/train_flaml/', views.train_flaml_baseline, name='train_flaml'),
    path('baseline/train_tune/', views.train_tune_baseline, name='train_tune'),
    path('baseline/train_autofolio/', views.train_autofolio_baseline, name='train_autofolio'),
    path('baseline/train_raha/', views.train_raha_baseline, name='train_raha'),
    path('baseline/status/', views.get_baseline_status, name='baseline_status'),
    path('baseline/data_split/', views.get_data_split_info, name='data_split_info'),
    path('baseline/test_set/', views.get_test_set_data, name='test_set_data'),
    
    # Recommend
    path('recommend/upload/', views.upload_inference, name='upload_inference'),
    path('recommend/features/', views.extract_recommend_features, name='recommend_features'),
    path('recommend/recommend/', views.get_recommendation, name='recommendation'),
    path('recommend/compare/', views.compare_baselines, name='compare_baselines'),
    path('recommend/impute/', views.run_imputation, name='imputation'),
    path('recommend/downstream/', views.run_downstream, name='downstream'),
    path('recommend/download/', views.download_result, name='download'),
    
    # Evaluation (High Priority - Implemented)
    path('recommend/setup_test_set/', views.setup_evaluation_from_test_set, name='setup_test_set'),
    path('recommend/setup_upload/', views.setup_evaluation_from_upload, name='setup_upload'),
    path('recommend/evaluation_status/', views.get_evaluation_status, name='evaluation_status'),
    
    # Evaluation Metrics (Medium Priority - NEW)
    path('recommend/compute_ground_truth_labels/', views.compute_ground_truth_labels, name='compute_ground_truth_labels'),
    path('recommend/evaluation_metrics/', views.get_evaluation_metrics, name='evaluation_metrics'),
    path('recommend/full_evaluation/', views.run_full_evaluation, name='full_evaluation'),
    
    # Celery (for feature extraction)
    path('task-status/<str:task_id>/', views.get_task_status, name='task_status'),
]
