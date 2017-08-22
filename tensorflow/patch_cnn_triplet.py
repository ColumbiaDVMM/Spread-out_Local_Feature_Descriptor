import tensorflow as tf

class PatchCNN_triplet:
    def __init__(self, CNNConfig):
        self.channel_num = CNNConfig["channel_num"]
        self.patch = tf.placeholder("float32", [None, CNNConfig["patch_size"], CNNConfig["patch_size"], CNNConfig["channel_num"]])
        self.patch_p = tf.placeholder("float32", [None, CNNConfig["patch_size"], CNNConfig["patch_size"], CNNConfig["channel_num"]])
        self.patch_n = tf.placeholder("float32", [None, CNNConfig["patch_size"], CNNConfig["patch_size"], CNNConfig["channel_num"]])

        self.margin_0 = CNNConfig["margin_0"]
        self.margin_1 = CNNConfig["margin_1"]
        self.margin_2 = CNNConfig["margin_2"]
        self.alpha = CNNConfig["alpha"]
        self.beta = CNNConfig["beta"]
        self.loss_type = CNNConfig["loss_type"]
        self.reg_type = CNNConfig["reg_type"]

        self.no_normalization = CNNConfig["no_normalization"]
        self.descriptor_dim = CNNConfig["descriptor_dim"]
        self._patch_size = CNNConfig["patch_size"]

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.model(self.patch)
            scope.reuse_variables()
            self.o2 = self.model(self.patch_p)
            scope.reuse_variables()
            self.o3 = self.model(self.patch_n)

        # Create loss
        if self.loss_type == 0:
            self.cost, self.eucd_p, self.cos_n1 = self.triplet_loss()
        elif self.loss_type == 1:
            self.cost, self.eucd_p, self.cos_n1 = self.N_pair_loss()
        elif self.loss_type == 2:
            self.cost, self.eucd_p, self.cos_n1 = self.LSSS_loss()

    def weight_variable(self, name, shape):
        weight = tf.get_variable(name = name+'_W', shape = shape, initializer = tf.random_normal_initializer(0, 0.1))
        return weight

    def bias_variable(self,name, shape):
        bias = tf.get_variable(name = name + '_b', shape = shape, initializer = tf.constant_initializer(0.1))
        return bias

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def conv2d_layer(self, name, shape, x):
        weight_init = tf.truncated_normal_initializer(stddev=.1)
        weight = tf.get_variable(name=name + '_W', dtype = tf.float32, shape=shape, initializer = weight_init)
        bias = tf.get_variable(name=name + '_b', dtype = tf.float32,
                               initializer = tf.constant(0.1, shape = [shape[3]], dtype = tf.float32))

        conv_val = tf.nn.relu(tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='VALID')+bias)
        return conv_val
    
    def conv2d_layer_BN(self, name, shape, x):
        weight_init = tf.truncated_normal_initializer(stddev=.1)
        weight = tf.get_variable(name=name + '_W', dtype = tf.float32, shape=shape, initializer = weight_init)
        bias = tf.get_variable(name=name + '_b', dtype = tf.float32, \
                               initializer = tf.constant(0.1, shape = [shape[3]], dtype = tf.float32))
        conv_val = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='VALID')+bias

        batch_mean2, batch_var2 = tf.nn.moments(conv_val,[0])
        scale2 = tf.get_variable(name = name+ '_bn_w', dtype = tf.float32, \
                               initializer = tf.constant(1, shape = [shape[3]], dtype = tf.float32))
        beta2 = tf.get_variable(name = name+ '_bn_b', dtype = tf.float32, \
                               initializer = tf.constant(0, shape = [shape[3]], dtype = tf.float32))
        bn_val = tf.nn.batch_normalization(conv_val,batch_mean2,batch_var2,beta2,scale2,1e-3)
        
        return tf.nn.relu(bn_val)

    def conv2d_layer_BN_with_padding(self, name, shape, x):
        weight_init = tf.truncated_normal_initializer(stddev=.1)
        weight = tf.get_variable(name=name + '_W', dtype = tf.float32, shape=shape, initializer = weight_init)
        bias = tf.get_variable(name=name + '_b', dtype = tf.float32, \
                               initializer = tf.constant(0.1, shape = [shape[3]], dtype = tf.float32))
        conv_val = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')+bias

        batch_mean2, batch_var2 = tf.nn.moments(conv_val,[0])
        scale2 = tf.get_variable(name = name+ '_bn_w', dtype = tf.float32, \
                               initializer = tf.constant(1, shape = [shape[3]], dtype = tf.float32))
        beta2 = tf.get_variable(name = name+ '_bn_b', dtype = tf.float32, \
                               initializer = tf.constant(0, shape = [shape[3]], dtype = tf.float32))
        bn_val = tf.nn.batch_normalization(conv_val,batch_mean2,batch_var2,beta2,scale2,1e-3)
        
        return tf.nn.relu(bn_val)

    def conv2d_layer_BN_with_stride(self, name, shape, x):
        weight_init = tf.truncated_normal_initializer(stddev=.1)
        weight = tf.get_variable(name=name + '_W', dtype = tf.float32, shape=shape, initializer = weight_init)
        bias = tf.get_variable(name=name + '_b', dtype = tf.float32, \
                               initializer = tf.constant(0.1, shape = [shape[3]], dtype = tf.float32))
        conv_val = tf.nn.conv2d(x, weight, strides=[1, 2, 2, 1], padding='SAME')+bias

        batch_mean2, batch_var2 = tf.nn.moments(conv_val,[0])
        scale2 = tf.get_variable(name = name+ '_bn_w', dtype = tf.float32, \
                               initializer = tf.constant(1, shape = [shape[3]], dtype = tf.float32))
        beta2 = tf.get_variable(name = name+ '_bn_b', dtype = tf.float32, \
                               initializer = tf.constant(0, shape = [shape[3]], dtype = tf.float32))
        bn_val = tf.nn.batch_normalization(conv_val,batch_mean2,batch_var2,beta2,scale2,1e-3)
        
        return tf.nn.relu(bn_val)

    def fc_layer_BN(self, name, shape, x):
        weight_init = tf.truncated_normal_initializer(stddev=.1)
        weight = tf.get_variable(name=name + '_W', dtype = tf.float32, shape=shape, initializer = weight_init)
        bias = tf.get_variable(name=name + '_b', dtype = tf.float32,\
                               initializer = tf.constant(0.1, shape = [shape[1]], dtype = tf.float32))

        fc_val = tf.matmul(x, weight)+bias
        batch_mean2, batch_var2 = tf.nn.moments(fc_val,[0])
        scale2 = tf.get_variable(name = name+ '_bn_w', dtype = tf.float32, \
                               initializer = tf.constant(1, shape = [shape[1]], dtype = tf.float32))
        beta2 = tf.get_variable(name = name+ '_bn_b', dtype = tf.float32, \
                               initializer = tf.constant(0, shape = [shape[1]], dtype = tf.float32))
        bn_val = tf.nn.batch_normalization(fc_val,batch_mean2,batch_var2,beta2,scale2,1e-3)
        return bn_val

    def _variable_with_weight_decay(self, name, shape, wd):
        dtype = tf.float32
        var = tf.get_variable(name=name, dtype = tf.float32, \
                shape=shape, initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def fc_layer(self, name, shape, x):
        weight_init = tf.truncated_normal_initializer(stddev=.1)
        weight = tf.get_variable(name=name + '_W', dtype = tf.float32, shape=shape, initializer = weight_init)
        bias = tf.get_variable(name=name + '_b', dtype = tf.float32,
                               initializer = tf.constant(0.1, shape = [shape[1]], dtype = tf.float32))

        fc_val = tf.matmul(x, weight)+bias
        return fc_val


    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

    def model(self, x):
        if self._patch_size == 32:
            h_conv1 = self.conv2d_layer_BN_with_padding('conv1', [3, 3, self.channel_num, 32], x)
            h_conv2 = self.conv2d_layer_BN_with_padding('conv2', [3, 3, 32, 32], h_conv1)
            h_conv3 = self.conv2d_layer_BN_with_stride('conv3', [3, 3, 32, 64], h_conv2)
            h_conv4 = self.conv2d_layer_BN_with_padding('conv4', [3, 3, 64, 64], h_conv3)
            h_conv5 = self.conv2d_layer_BN_with_stride('conv5', [3, 3, 64, 128], h_conv4)
            h_conv6 = self.conv2d_layer_BN_with_padding('conv6', [3, 3, 128, 128], h_conv5)
            h_conv7 = self.conv2d_layer('conv7', [8, 8, 128, 128], h_conv6)
            output = tf.reshape(h_conv7, [-1, 128])
            
        elif self._patch_size == 64:
            if self.no_normalization:
                h_conv1 = self.conv2d_layer_BN('conv1', [7, 7, self.channel_num, 32], x)
                h_pool1 = self.max_pool_2x2(h_conv1)
                h_conv2 = self.conv2d_layer_BN('conv2', [6, 6, 32, 64], h_pool1)
                h_pool2 = self.max_pool_2x2(h_conv2)
                h_conv3 = self.conv2d_layer_BN('conv3', [5, 5, 64, 128], h_pool2)
                h_pool3 = self.max_pool_2x2(h_conv3)
                pool3_flatten = tf.reshape(h_pool3, [-1, 4*4*128])
                output = self.fc_layer('fc1',[4*4*128,self.descriptor_dim],pool3_flatten)
            else:
                h_conv1 = self.conv2d_layer_BN('conv1', [7, 7, self.channel_num, 32], x)
                h_pool1 = self.max_pool_2x2(h_conv1)
                h_conv2 = self.conv2d_layer_BN('conv2', [6, 6, 32, 64], h_pool1)
                h_pool2 = self.max_pool_2x2(h_conv2)
                h_conv3 = self.conv2d_layer_BN('conv3', [5, 5, 64, 128], h_pool2)
                h_pool3 = self.max_pool_2x2(h_conv3)
                pool3_flatten = tf.reshape(h_pool3, [-1, 4*4*128])
                output = self.fc_layer('fc1',[4*4*128,self.descriptor_dim],pool3_flatten)
        else:
            output = []
    
        if self.no_normalization:
            return output/30.0
        else:
            return tf.nn.l2_normalize(output,dim=1)

    def triplet_loss(self):
        margin_0 = self.margin_0
        margin_1 = self.margin_1
        margin_2 = self.margin_2
        alpha = self.alpha

        d_over_1 = tf.constant(self.beta/self.descriptor_dim)

        eucd_p = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd_p = tf.reduce_sum(eucd_p, 1)
        eucd_p = tf.sqrt(eucd_p+1e-6)

        eucd_n1 = tf.pow(tf.subtract(self.o1, self.o3), 2)
        eucd_n1 = tf.reduce_sum(eucd_n1, 1)
        eucd_n1 = tf.sqrt(eucd_n1+1e-6)

        eucd_n2 = tf.pow(tf.subtract(self.o2, self.o3), 2)
        eucd_n2 = tf.reduce_sum(eucd_n2, 1)
        eucd_n2 = tf.sqrt(eucd_n2+1e-6)

        secMoment_n1 = tf.pow(tf.reduce_sum(tf.multiply(self.o1, self.o3),1),2)
        secMoment_n2 = tf.pow(tf.reduce_sum(tf.multiply(self.o2, self.o3),1),2)
        
        mean =  tf.pow(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.o1, self.o3),1)),2)

        random_negative_margin = tf.constant(margin_0)
        alpha_tf = tf.constant(alpha)
        
        positive_margin = tf.constant(margin_1)
        
        with tf.name_scope('all_loss'):
            #invertable loss for standard patches
            with tf.name_scope('rand_neg'):
                rand_neg = tf.reduce_mean(tf.maximum(secMoment_n1,secMoment_n2))
                #rand_neg = tf.reduce_mean(secMoment_n1)
            #covariance loss for transformed patches
            with tf.name_scope('pos'):
                pos = tf.maximum(tf.subtract(positive_margin,tf.subtract(eucd_n1,eucd_p)), 0)
            #total loss
            with tf.name_scope('loss'):
                losses = pos
                if self.reg_type == 0:
                    loss = tf.reduce_mean(losses) + tf.multiply(alpha_tf, \
                        mean + tf.maximum(tf.subtract(rand_neg,d_over_1),0))
                elif self.reg_type == 1:
                    loss = tf.reduce_mean(losses) + tf.multiply(alpha_tf, \
                        mean + tf.abs(tf.subtract(rand_neg,d_over_1)))
                elif self.reg_type == 2:
                    loss = tf.reduce_mean(losses) + tf.multiply(alpha_tf, \
                        mean + tf.sqrt(tf.abs(tf.subtract(rand_neg,d_over_1))))
                else:
                    loss = tf.reduce_mean(losses) +  \
                        mean + tf.multiply(alpha_tf,tf.pow(tf.subtract(rand_neg,d_over_1),2))

        tf.summary.scalar('random_negative_loss', rand_neg)
        tf.summary.scalar('positive_loss', tf.reduce_mean(pos))
        tf.summary.scalar('total_loss', loss)

        return loss, eucd_p, secMoment_n1


    def N_pair_loss(self):
        alpha = self.alpha
        beta = self.beta

        margin_1 = self.margin_1

        neg_val = tf.exp(tf.reduce_sum(self.o1*self.o3,1) + margin_1)
        
        pos_val = tf.reduce_sum(self.o1*self.o2,1)
        pos_matrix = tf.tile(pos_val,[tf.size(pos_val)])
        pos_matrix = tf.reshape(pos_matrix,[tf.size(pos_val),tf.size(pos_val)])
        
        neg_matrix = tf.matmul(self.o1,tf.transpose(self.o2))
        
        eucd_p = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd_p = tf.reduce_sum(eucd_p, 1)
        eucd_p = tf.sqrt(eucd_p+1e-6)
        
        d_over_1 = tf.constant(beta/self.descriptor_dim)

        cos_n1 = tf.pow(tf.reduce_sum(tf.multiply(self.o1, self.o3),1),2)
        cos_n2 = tf.pow(tf.reduce_sum(tf.multiply(self.o2, self.o3),1),2)
        
        mean = tf.pow(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.o1, self.o3),1)),2)

        norm_1 = tf.reduce_mean(tf.reduce_sum(tf.pow(self.o1, 2),1))
        norm_2 = tf.reduce_mean(tf.reduce_sum(tf.pow(self.o2, 2),1))
        norm_3 = tf.reduce_mean(tf.reduce_sum(tf.pow(self.o3, 2),1))
        
        with tf.name_scope('all_loss'):
            #invertable loss for standard patches
            with tf.name_scope('pos_val'):
                mean_pos_val = tf.reduce_mean(pos_val) 
            with tf.name_scope('neg_val'):
                mean_neg_val = tf.reduce_mean(neg_val) 
            with tf.name_scope('npair_loss'):
                npair_loss = tf.reduce_mean(tf.log(tf.exp(tf.reduce_sum(tf.maximum(neg_matrix + margin_1 - pos_matrix,0),1))))
            with tf.name_scope('norm'):
                mean_norm = tf.reduce_mean(norm_1)
            with tf.name_scope('rand_neg'):
                rand_neg = tf.reduce_mean(tf.maximum(cos_n1,cos_n2))
            #total loss
            with tf.name_scope('loss'):
                loss = npair_loss +  alpha*(mean + tf.maximum(tf.subtract(rand_neg,d_over_1),0)) #\
                 #+ beta * (norm_1 + norm_2 + norm_3)
                
        tf.summary.scalar('positive_val', mean_pos_val)
        tf.summary.scalar('negative_val', mean_neg_val)
        tf.summary.scalar('npair_loss', npair_loss)
        tf.summary.scalar('random_negative_loss', rand_neg)
        tf.summary.scalar('total_loss', loss)
        tf.summary.scalar('norm', mean_norm)

        return loss, eucd_p, norm_1
