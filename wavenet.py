
def create_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable

def create_bias_variable(name, shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)

def create_variables(use_initialfilter, initial_filter_width, 
    initial_channels, residual_channels, dilations, filter_width, 
    dilation_channels, skip_channels, use_bias):
    var = dict()
    with tf.variable_scope('wavenet'):
        #initial layer
        if use_initialfilter:
            with tf.variable_scope('causal_layer'):
                layer = dict()
                layer['filter'] = create_variable(
                    'filter', 
                    [initial_filter_width, initial_channels, residual_channels])
            var['causal_layer'] = layer
        #dilated layer
        var["dilated_stack"] = list()
        with tf.variable_scope('dilated_stack'):
            for i, dilation in enumerate(dilations):
                with tf.variable_scope('layer{}'.format(i)):
                    layer = dict()
                    layer["filter"] = create_variable(
                        'filter', 
                        [filter_width, residual_channels, dilation_channels])
                    layer["gate"] = create_variable(
                        'filter', 
                        [filter_width, residual_channels, dilation_channels])
                    layer["dense"] = create_variable(
                        'dense',
                        [1, dilation_channels, residual_channels])
                    layer['skip'] = create_variable(
                        'skip',
                        [1, dilation_channels, skip_channels])
                    #if use bias
                    if use_bias:
                        layer['filter_bias'] = create_bias_variable(
                            'filter_bias',
                            [dilation_channels])
                        layer['gate_bias'] = create_bias_variable(
                            'gate_bias',
                            [dilation_channels])
                        layer['dense_bias'] = create_bias_variable(
                            'dense_bias',
                            [residual_channels])
                        layer['skip_bias'] = create_bias_variable(
                            'slip_bias',
                            [skip_channels])
                    var['dilated_stack'].append(layer)
        #1*1 conv net in the end
        with tf.variable_scope("postprocessing"):
            layer = dict()
            layer["postprocess1"] = create_variable(
                "postprocess1",
                [1, skip_channels, skip_channels])
            layer["postprocess2"] = create_variable(
                "postprocess2",
                [1, skip_channels, initial_channels])
            if use_bias:
                layer['postprocess1_bias'] = create_bias_variable(
                    'postprocess1_bias',
                    [skip_channels])
                layer['postprocess2_bias'] = create_bias_variable(
                    'postprocess2_bias',
                    [initial_channels])            
            var['postprocessing'] = layer
    return var


def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])


def causal_conv(value, filter_, dilation, name='causal_conv'):
    with tf.name_scope(name):
        filter_width = tf.shape(filter_)[0]
        if dilation > 1:
            transformed = time_to_batch(value, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1,
                                padding='VALID')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
        # Remove excess elements at the end.
        out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, out_width, -1])
        return result


def create_causal_layer(variables, input_batch):
    weights_filter = variables['causal_layer']['filter']
    return causal_conv(input_batch, weights_filter, 1)


def create_dilation_layer(variables, use_bias, input_batch, layer_index, dilation, output_width):
    weights_filter = variables['filter']
    weights_gate = variables['gate']
    #
    conv_filter = causal_conv(input_batch, weights_filter, dilation)
    conv_gate = causal_conv(input_batch, weights_gate, dilation)
    if use_bias:
        filter_bias = variables['filter_bias']
        gate_bias = variables['gate_bias']
        conv_filter = tf.add(conv_filter, filter_bias)
        conv_gate = tf.add(conv_gate, gate_bias)
    #
    out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)
    weights_dense = variables['dense']
    transformed = tf.nn.conv1d(out, weights_dense, stride = 1, padding = "SAME", name = "dense")
    #
    skip_cut = tf.shape(out)[1] - output_width
    out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
    weights_skip = variables['skip']
    skip_contribution = tf.nn.conv1d(out_skip, weights_skip, stride = 1, padding = "SAME", name = 'skip')
    if use_bias:
        dense_bias = variables['dense_bias']
        skip_bias = variables['skip_bias']
        transformed = transformed + dense_bias
        skip_contribution = skip_contribution + skip_bias
    #
    input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
    input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])
    return skip_contribution, input_batch + transformed


def create_network(input_batch, initial_channels, filter_width, dilations, variables, use_bias, receptive_field):
    with tf.name_scope('wavenet'):
        network_input_width = tf.shape(input_batch)[1] - 1
        input_batch = tf.slice(input_batch, [0, 0, 0],
                                     [-1, network_input_width, -1])
        outputs = []
        output_width = tf.shape(input_batch)[1] - receptive_field + 1
        current_layer = input_batch
        #initial conv layer
        if use_initialfilter:
            with tf.name_scope('causal_layer'):
                current_layer = create_causal_layer(variables, current_layer)
        #dilation layer
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(dilations):
                with tf.name_scope('layer{}'.format(layer_index)):            
                    output, current_layer = create_dilation_layer(
                        variables['dilated_stack'][layer_index], use_bias, 
                        current_layer, layer_index, 
                        dilation, output_width)
                    outputs.append(output)
        #
        with tf.name_scope('postprocessing'):
            w1 = variables['postprocessing']['postprocess1']
            w2 = variables['postprocessing']['postprocess2']
            if use_bias:
                b1 = variables['postprocessing']['postprocess1_bias']
                b2 = variables['postprocessing']['postprocess2_bias']
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride = 1, padding = "SAME")
            if use_bias:
                conv1 = tf.add(conv1, b1)
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, w2, stride = 1, padding = "SAME")
            if use_bias:
                conv2 = tf.add(conv2, b2)
    return conv2



def loss(encoded, raw_output, initial_channels, receptive_field):
    with tf.name_scope('loss'):
        target_output = tf.slice(encoded,
                    [0, receptive_field, 0],
                    [-1, -1, -1])
        target_output = tf.reshape(target_output,
                                   [-1, initial_channels])
        prediction = tf.reshape(raw_output,
                                [-1, initial_channels])
        loss = tf.losses.mean_squared_error(prediction, prediction)
        reduced_loss = tf.reduce_mean(loss)
        return reduced_loss



def main(sample_length, use_bias, use_initialfilter,
    initial_filter_width, initial_channels,
    dilations, residual_channels, 
    dilation_channels, skip_channels, 
    filter_width, learning_rate):
    with tf.name_scope('wavenet'):
        input_batch = tf.placeholder(tf.float32, shape = [None, sample_length, initial_channels])

    variables = create_variables(use_initialfilter, initial_filter_width, 
    initial_channels, residual_channels, dilations, filter_width, 
    dilation_channels, skip_channels, use_bias)

    receptive_field = ((filter_width - 1) * sum(dilations) + 1 + filter_width - 1)

    raw_output = create_network(input_batch, initial_channels, filter_width, dilations, variables, use_bias, receptive_field)
    loss = loss(input_batch, raw_output, initial_channels, receptive_field)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)
    trainable = tf.trainable_variables()

    optim = optimizer.minimize(loss, var_list=trainable)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)


    

    for step in range(1000):
        r_idx = sample(range(len(df_train) - 7200),1)[0]
        ts = df_train.iloc[r_idx:r_idx+7200,:].reshape([1,7200, -1])
        for t in range(10):
            loss_value, _ = sess.run([loss, optim])
        loss_list.append(loss_value)
        if len(loss_list)>50:
            print(np.average(loss_list[-50:]))




if __name__ == "__main__":

    use_bias = True
    use_initialfilter = True
    initial_filter_width = 2
    initial_channels = 10######10
    dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    residual_channels = 32
    dilation_channels = 32
    skip_channels = 512
    filter_width = 2
    learning_rate = 0.00001
    sample_length = 7200

    main(use_bias, use_initialfilter, initial_filter_width, initial_channels,
        dilations, residual_channels, 
        dilation_channels, skip_channels, filter_width, learning_rate)

