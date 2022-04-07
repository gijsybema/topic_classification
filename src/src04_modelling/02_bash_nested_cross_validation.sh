#!/bin/bash
python3 src/src04_modelling/02-nested_cross_validation.py afhandeling_proces nb_count rf_count rf_tfidf svm_tfidf svm_count 
python3 src/src04_modelling/02-nested_cross_validation.py algemeen lr_count rf_count svm_count svm_tfidf 
python3 src/src04_modelling/02-nested_cross_validation.py dig_account svm_count ada_count ada_tfidf gb_count gb_tfidf 
python3 src/src04_modelling/02-nested_cross_validation.py dig_contact_web gb_count ada_count svm_count 
python3 src/src04_modelling/02-nested_cross_validation.py dig_func_web svm_count ada_count ada_tfidf gb_tfidf 
python3 src/src04_modelling/02-nested_cross_validation.py dig_gebruik_web lr_count svm_count ada_count ada_tfidf gb_tfidf 
python3 src/src04_modelling/02-nested_cross_validation.py digitaal svm_count ada_count ada_tfidf gb_count gb_tfidf 
python3 src/src04_modelling/02-nested_cross_validation.py houding_gedrag lr_count rf_count svm_count ada_count ada_tfidf 
python3 src/src04_modelling/02-nested_cross_validation.py info gb_tfidf svm_count rf_count ada_tfidf ada_count 
python3 src/src04_modelling/02-nested_cross_validation.py kennis_vaardigheden ada_count ada_tfidf gb_count svm_count lr_count
python3 src/src04_modelling/02-nested_cross_validation.py kv_advies ada_tfidf ada_count gb_tfidf gb_count 
python3 src/src04_modelling/02-nested_cross_validation.py kv_deskundig ada_count svm_count ada_tfidf lr_count 
python3 src/src04_modelling/02-nested_cross_validation.py prijs_kwaliteit gb_count svm_tfidf svm_count rf_count nb_count 
python3 src/src04_modelling/02-nested_cross_validation.py telefoon_contact rf_count rf_tfidf gb_tfidf ada_tfidf svm_tfidf 