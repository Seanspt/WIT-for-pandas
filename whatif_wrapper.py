def whatif_tool(df, feature_cols, predict_func, compare_predict_func=None, label_vocab=None, max_n_rows=1000):
    """
    This is a what-if tool wrapper for non-tensorflow users. If you are a tensorflow user,
    just follow the instruction on the official website and have fun.
    
    Args:
        df:           A pandas dataframe to analyse.
        feature_cols: A list of string, cols in which will be used as features.
        predict_func: Predict function to be performed on df. Usually, the data 
                      preprocessing procedure are embedded in-side by a closure.
        compare_predict_func: Same as predict_func. Used to compare with the result 
                      produced by `predict_func`.
        label_vocab:  Label names for display.
        max_n_rows:   max row limit to protect your browser, ^-^.
    """
    if df.shape[0] > max_n_rows:
        raise Exception("max row limit exceeded. Expected no more than %d rows, %d got." % (max_n_rows, df.shape[0]))
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import functools

    # Creates a tf feature spec from the dataframe and columns specified.
    def create_feature_spec(df, columns=None):
        feature_spec = {}
        if columns == None:
            columns = df.columns.values.tolist()
        for f in columns:
            if df[f].dtype is np.dtype(np.int64):
                feature_spec[f] = tf.FixedLenFeature(shape=(), dtype=tf.int64)
            elif df[f].dtype is np.dtype(np.float64):
                feature_spec[f] = tf.FixedLenFeature(shape=(), dtype=tf.float32)
            else:
                feature_spec[f] = tf.FixedLenFeature(shape=(), dtype=tf.string)
        return feature_spec

    # Creates simple numeric and categorical feature columns from a feature spec and a
    # list of columns from that spec to use.
    #
    # NOTE: Models might perform better with some feature engineering such as bucketed
    # numeric columns and hash-bucket/embedding columns for categorical features.
    def create_feature_columns(columns, feature_spec):
        ret = []
        for col in columns:
            if feature_spec[col].dtype is tf.int64 or feature_spec[col].dtype is tf.float32:
                ret.append(tf.feature_column.numeric_column(col))
            else:
                ret.append(tf.feature_column.indicator_column(
                    tf.feature_column.categorical_column_with_vocabulary_list(col, list(df[col].unique()))))
        return ret

    # An input function for providing input to a model from tf.Examples
    def tfexamples_input_fn(examples, feature_spec, label, mode=tf.estimator.ModeKeys.EVAL,
                           num_epochs=None, 
                           batch_size=64):
        def ex_generator():
            for i in range(len(examples)):
                yield examples[i].SerializeToString()
        dataset = tf.data.Dataset.from_generator(
          ex_generator, tf.dtypes.string, tf.TensorShape([]))
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example, label, feature_spec))
        dataset = dataset.repeat(num_epochs)
        return dataset

    # Parses Tf.Example protos into features for the input function.
    def parse_tf_example(example_proto, label, feature_spec):
        parsed_features = tf.parse_example(serialized=example_proto, features=feature_spec)
        target = parsed_features.pop(label)
        return parsed_features, target

    # Converts a dataframe into a list of tf.Example protos.
    def df_to_examples(df, columns=None):
        examples = []
        if columns == None:
            columns = df.columns.values.tolist()
        for index, row in df.iterrows():
            example = tf.train.Example()
            for col in columns:
                if df[col].dtype is np.dtype(np.int64):
                    example.features.feature[col].int64_list.value.append(int(row[col]))
                elif df[col].dtype is np.dtype(np.float64):
                    example.features.feature[col].float_list.value.append(row[col])
                elif row[col] == row[col]:
                    example.features.feature[col].bytes_list.value.append(row[col].encode('utf-8'))
            examples.append(example)
        return examples

    tool_height_in_px = 1000  #@param {type: "number"}
    feature_spec = create_feature_spec(df, feature_cols)
    def make_predict_func(predict_func, cols):
        def _predict_func(examples):
            data_dict = {}
            for c in cols:
                data_dict[c] = []
            iterator = tf.data.Dataset.from_tensor_slices(tf.parse_example([ex.SerializeToString() for ex in test_examples], feature_spec)).make_one_shot_iterator()
            ex = iterator.get_next()
            with tf.Session() as sess:
                while True:
                    try:
                        ex_a = sess.run(ex)
                        for c in cols:
                            data_dict[c].append(ex_a[c] if not isinstance(ex_a[c], bytes) else ex_a[c].decode("utf-8"))
                    except tf.errors.OutOfRangeError:
                        break
            df = pd.DataFrame(data_dict)
            return predict_func(df[cols])
        return _predict_func
    from witwidget.notebook.visualization import WitConfigBuilder
    from witwidget.notebook.visualization import WitWidget


    test_examples = df_to_examples(df)
    a_predict_func = make_predict_func(predict_func, feature_cols)
    # Setup the tool with the test examples and the trained classifier
    config_builder = WitConfigBuilder(test_examples).set_custom_predict_fn(a_predict_func)
    if label_vocab:
        config_builder = config_builder.set_label_vocab(label_vocab)
    if compare_predict_func:
        a_compare_predict_func = make_predict_func(compare_predict_func, feature_cols)
        config_builder = config_builder.set_compare_custom_predict_fn(a_compare_predict_func)
    return WitWidget(config_builder, height=tool_height_in_px)
    