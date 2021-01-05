classdef myMN_NB
    methods(Static)

        %% find mean and standard deviation of all features for each class label in training example
        %% get prior probability of each class label (will be used in for posterior prob later)
        %% returned trained model
        function m = fit(train_examples, train_labels)
            %% find all classes in training example
            m.unique_classes = unique(train_labels);
            %% find length of array of unique classes in to use as number of interation in for loop to populate mean and standard dev cells
            m.n_classes = length(m.unique_classes);
            m.training_points = size(train_examples,1);
            m.train_labels = train_labels;
            
            %% initialize prior probabilty array
            m.priors = [];
            % start prior probability population
            %% estimate how likely a label is to occure based on how many labels exist in set
            %% for all possible class labels in dataset calculate the prior probability of every training data associated with said class label
            for i = 1:m.n_classes
                %% copy current class label in this_class
				this_class = m.unique_classes(i);
                %% fetch all training examples for current class and copy them in examples_from_this_class
                examples_from_this_class = train_examples{train_labels==this_class,:};
                %% calculate the probability that a randomly chosen example in examples_from_this_class is likely to be part of current class 
                m.priors(end+1) = vpa(size(examples_from_this_class,1) / size(train_labels,1));
            
			end
            % end prior probability population
            
            %% begin word to matrix transformation
            %% this matrix will be of 1xN where N is the number of unique
            %% words in our training set (features of the corpus)
            %% call dictionnary of stops words
            stopwords = StopWords.stop_words();
            % convert our data points to string array of tx1 where t is 
            % a line of data (1 training example/document/headline) 
            All_dox_original_string_array = train_examples{:,:};
            
            % transform concatenated data into a dictionnary of unique
            % words
            [words_dictionary,Nmbrs, indicies_freqncy_of_wrds]= myMN_NB.transform_text(All_dox_original_string_array, stopwords);
                                    
            % create a final matrix of 1 x num_of_features
            matrix_model = zeros(1,size(words_dictionary,2));
            
            % save features matrix and features count
            m.feature_matrix = matrix_model;
            m.num_of_features = size(matrix_model,2);
            % log all unique word in our model
            m.unique_words_dictionary = words_dictionary;
            
            %% populate our feature matrix with training examples to
            %% allow mean and standard deviation calculation of each training
            %% example (will be usefull in classification phase)    
            feature_matrix = myMN_NB.populate_feature_matrix(train_examples, matrix_model, StopWords.stop_words(),m);
            % save populate feature matrix with frequencies
            m.data_matrix = feature_matrix;
            % save frequencies of all features in corpus
            m.aggregated_feature_frequency = sum(feature_matrix);
            m.SIZE = size(m.aggregated_feature_frequency,2);

        end
        
        %% transform a data point (string input from table) into 1xY matrix where Y is the number of unique words in corpus
        
        function [words_dictionary_, Nmbrs, freqncy_of_wrds] = transform_text(data_string, stopwords)         
            
            %transform data into tokens
            documents = tokenizedDocument(lower(data_string));
            
            % erase all punctions they are not required for classification            
            cleaned_input = erasePunctuation(documents); %chnage back to cleaned input
                        
 
            % stem the string array (reduce all words to their root form) this will reduce compute time and feature count 
            cleaned_input = normalizeWords(cleaned_input, 'Style',"stem");
            
            
            words = [stopWords, stopwords.words];
            
            %remove stop words
            cleaned_input = removeWords(cleaned_input, words);  
  
            % split into tokens
            cleaned_input = split(strjoin(joinWords(cleaned_input))," ");

            cleaned_input = eraseTags(cleaned_input);
            cleaned_input = eraseURLs(cleaned_input);
             
            
            cleaned_input(cleaned_input(:,1)=='') = [];
            
            %data_string = lower(erasePunctuation(data_string));
            Nmbrs = ["0" "1" "2" "3" "4" "5" "6" "7" "8" "9"];
            % erase all Numeric chars
            cleaned_input = erase(cleaned_input, Nmbrs);
            
            % find frequency of important words
            [words_dictionary, indicies_uniq_wrds, freqncy_of_wrds] = unique(cleaned_input);
            % reshape matrix for later use
            mySize = length(words_dictionary);
            words_dictionary_ = reshape(words_dictionary, 1, mySize);
            
        end
        
        %% populate our feature matrix of 1xt where t is the number of features (unique words in our matrix)
        %% return a populated matrix of 1xt
        function feature_matrix = populate_feature_matrix(Docx_string, matrix_model, stopwords, m)
            
                %% populate our feature matrix with training examples to
                %% allow mean and standard deviation calculation of each training
                %% example (will be usefull in classification phase)
                feature_matrix = matrix_model;
                % for every training example (document/headline)
                for p = 1:size(Docx_string,1)
                   [unique_words2, Nmbrs, freqncy_of_wrds2] = myMN_NB.transform_text(Docx_string{p,:}, stopwords);
                     
                   % tally up frequencies
                   tally = accumarray(freqncy_of_wrds2, 1);
                   %unique_words2__
                   % find rows in word dictionary (feature dictionary) corresponding to unique words of current observation 
                   [R1_, CL1_] = find([unique_words2(1,:)]==m.unique_words_dictionary(:));
                   
                   % index matrix with frequencies of found unique words for
                   % current example
                   feature_matrix(p,R1_) = tally(CL1_,:);
                   
                end 
                
        end

        %% Compute a likelihood given each class, for the new example we are trying to classify
        %% Multiply each likelihood by the prior for the corresponding class to give a value proportional to the posterior probability
        %% Generate a prediction of the class label by finding the class with the largest resulting value
        %% return a categorical array of predictions  
        function predictions = predict(m, test_examples)
            %% initialize categorical array
            predictions = categorical;
            %% for every row of test data 
            for i=1:size(test_examples,1)

				fprintf('classifying example %i/%i\n', i, size(test_examples,1));
                %% copy test example at current row index
                this_test_example = test_examples{i,:};

                %% call dictionnary of stops words
                stopwords = StopWords.stop_words();
                % transform data into a dictionnary of unique
                % words                
                [words_dictionary,Nmbrs, indicies_freqncy_of_wrds] = myMN_NB.transform_text(this_test_example, stopwords);
                
                % tally up frequencies
                tally = accumarray(indicies_freqncy_of_wrds, 1);
                % find rows in word dictionary (feature dictionary) corresponding to unique words of current observation 
                [R1_, L] = find([words_dictionary(1,:)]==m.unique_words_dictionary(:)); %%correct this
                
                % create a dummy matrix of 1 x num_of_features in training
                % data
                dummy_matrix = zeros(1,m.num_of_features);
                
                if ~isempty(R1_)
                    % index dummy matrix with frequencies of unique words
                    % in found in current test point at exact feature point
                    dummy_matrix(:,R1_) = tally(L,:);
                end
                %% calculate the likelihood of current example to classify, using each class label stored in (m.priors)
                this_prediction = myMN_NB.predict_one(m, dummy_matrix, words_dictionary(:,L));
                %% add prediction to predictions array
                predictions(end+1) = this_prediction;
            
			end
        end
        
        %% Compute a likelihood given each class, for the new example we are trying to classify
        %% calculate prior probability of test example and compute a class label prediction 
        %% return prediction 
        function prediction = predict_one(m, this_test_example, words_dictionary)
            %% for every class in our training data set
            for i=1:m.n_classes
                %% copy current class label in this_class
				this_class = m.unique_classes(i);
                %% initialize a dummy matrix of num of data associated with
                %% this label in data set by num of features in data set                
                matrix_this_class = zeros(size(m.data_matrix(m.train_labels==this_class,:),1),m.num_of_features);
                matrix_this_class = m.data_matrix(m.train_labels==this_class,:);
                
                %% find likelihood of current test example, look at each feature value of test example as an independant event (class conditional indepence of naive bayes)
				this_likelihood = myMN_NB.calculate_likelihood(m, this_test_example, i, matrix_this_class, words_dictionary);
                %% get prior probability of current class copy in this_prior
                this_prior = myMN_NB.get_prior(m, i);
                
                %% calculate and save posterior probability (likelihood * prior probability) of class attached to test example
                posterior_(i) = vpa(this_likelihood * this_prior);
            
            end
            %% find the most likely class label(maximum value) in posterior_ array
            [winning_value_, winning_index] = max(posterior_);

            %% use index of winning_value which is the same as index of class in unique_classes array
            prediction = m.unique_classes(winning_index);

        end
        
        %% find likelihood of current test example, since naives bayes can't capture relationships between features,
        %% look at each feature value of test example as an independant event (class conditional indepence of naive bayes)
        function likelihood = calculate_likelihood(m, this_test_example, class, matrix_this_class, words_dictionary)
            
			likelihood = vpa(1);
            
            
            %% for every feature (Word in memorized dictionary)
			for feature_index=1:size(this_test_example,2)
                %% since naives bayes can't capture relationships between features, 
                %% we need to capture probability distribution feature by feature and multiply all feature pd together
                % power raise will be used to raise the probability of
                % current feature to the amount of it's frequency. no point
                % in computing same probability twice        
                % allows to keep identical feature matrix size to training
                % corpus

                frequency_power_raise = 1;
                % if current feature is not zero use it's frequency as
                
                if this_test_example (:,feature_index) > 0
                    frequency_power_raise = this_test_example(:,feature_index);      
                else
                    % no point in calculating a feature that doesn't exist
                    % in our test document, it would only reduce accuracy
                    % and increase execution time
                    continue;                  
                end
                
                % compute likelihood of feature  
                likelihood = likelihood * (myMN_NB.laplace_smoothing(m, feature_index, matrix_this_class)^frequency_power_raise);
            end
        end
        

        %% get prior probability in model (m) using class index (class)
        %% return prior probability fraction
        function prior = get_prior(m, class)
            %% get prior probability index from model using class index 
			prior = m.priors(class);        
        end
        
        %% laplace smoothing willl get read of all zeroes likelihood as they
        %% will nullify our probabily, it helps avoid overfitting as there
        %% is no way we can see every possible word in existance
        
        function lp_smoothed = laplace_smoothing( m, feature_index, matrix_this_class)
            
            
            alpha = 1;
            %frequency of current feature in this class 
            %matrix_this_class;
            this_feature_frequency_this_class = sum(matrix_this_class(:,feature_index));
            % number of unique words (features)
            cardinality_training_corpus = m.num_of_features;
            % Unique words observed in this class
            Total_uniq_wrds_observations_this_class = sum(sum(matrix_this_class));
            
            
            % apply smoothing  
            part1 = this_feature_frequency_this_class + alpha;
            part2 = Total_uniq_wrds_observations_this_class + (cardinality_training_corpus * alpha);
            %use vpa to keep extreme precision of value since we are
            %working with really small floats
            lp_smoothed = vpa(part1/part2);
            
        end       
        
        
    end

end

