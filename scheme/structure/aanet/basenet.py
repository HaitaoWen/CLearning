from scheme.base import *
import torch.nn.functional as F
from scheme.replay.podnet.classifiers import Classifier, CosineClassifier, DomainClassifier, MCCosineClassifier
from scheme.replay.podnet.postprocessors import FactorScalar, HeatedUpScalar, InvertedFactorScalar


class BasicNet(nn.Module):

    def __init__(
        self,
        convnet_1,
        convnet_2,
        classifier_kwargs={},
        postprocessor_kwargs={},
        wordembeddings_kwargs={},
        init="kaiming",
        device=None,
        return_features=False,
        extract_no_act=False,
        classifier_no_act=False,
        attention_hook=False,
        rotations_predictor=False,
        gradcam_hook=False,
        dataset=None,
    ):
        super(BasicNet, self).__init__()

        if postprocessor_kwargs.get("type") == "learned_scaling":
            self.post_processor = FactorScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") == "inverted_learned_scaling":
            self.post_processor = InvertedFactorScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") == "heatedup":
            self.post_processor = HeatedUpScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") is None:
            self.post_processor = None
        else:
            raise NotImplementedError(
                "Unknown postprocessor {}.".format(postprocessor_kwargs["type"])
            )
        printlog("Post processor is: {}".format(self.post_processor))

        self.convnet_1 = convnet_1
        self.convnet_2 = convnet_2
        self.dataset = dataset

        if "type" not in classifier_kwargs:
            raise ValueError("Specify a classifier!", classifier_kwargs)
        if classifier_kwargs["type"] == "fc":
            self.classifier = Classifier(self.convnet_1.out_dim, **classifier_kwargs)
        elif classifier_kwargs["type"] == "cosine":
            self.classifier = CosineClassifier(
                self.convnet_1.out_dim, **classifier_kwargs
            )
        elif classifier_kwargs["type"] == "mcdropout_cosine":
            self.classifier = MCCosineClassifier(
                self.convnet_1.out_dim, **classifier_kwargs
            )
        else:
            raise ValueError("Unknown classifier type {}.".format(classifier_kwargs["type"]))

        self.rotations_predictor = None
        self.word_embeddings = None
        self.return_features = return_features
        self.extract_no_act = extract_no_act
        self.classifier_no_act = classifier_no_act
        self.attention_hook = attention_hook
        self.gradcam_hook = gradcam_hook
        self.device = device

        self.domain_classifier = None

        if self.gradcam_hook:
            self._hooks = [None, None]
            printlog("Setting gradcam hook for gradients + activations of last conv.")
            self.set_gradcam_hook()
        if self.extract_no_act:
            printlog("Features will be extracted without the last ReLU.")
        if self.classifier_no_act:
            printlog("No ReLU will be applied on features before feeding the classifier.")

        self.to(self.device)

    def on_task_end(self):
        if isinstance(self.classifier, nn.Module):
            self.classifier.on_task_end()
        if isinstance(self.post_processor, nn.Module):
            self.post_processor.on_task_end()

    def on_epoch_end(self):
        if isinstance(self.classifier, nn.Module):
            self.classifier.on_epoch_end()
        if isinstance(self.post_processor, nn.Module):
            self.post_processor.on_epoch_end()

    def forward_convnet(self, x):
        if self.dataset == 'CIFAR100':
            return self.forward_convnet_cifar100(x)
        elif self.dataset in ('ImageNet100', 'ImageNet1000'):
            return self.forward_convnet_imgnet(x)
        else:
            import pdb
            pdb.set_trace()

    def forward_convnet_imgnet(self, x):

        x_1 = self.convnet_1.conv1(x)
        x_1 = self.convnet_1.bn1(x_1)
        x_1 = self.convnet_1.relu(x_1)
        x_1 = self.convnet_1.maxpool(x_1)
        x_1 = self.convnet_1.layer1(x_1)

        x_2 = self.convnet_2.conv1(x)
        x_2 = self.convnet_2.bn1(x_2)
        x_2 = self.convnet_2.relu(x_2)
        x_2 = self.convnet_2.maxpool(x_2)
        x_2 = self.convnet_2.layer1(x_2)

        x_new_1 = 0.5*x_1 + 0.5*x_2

        level2_x_1 = self.convnet_1.layer2(self.convnet_1.end_relu(x_new_1))
        level2_x_2 = self.convnet_2.layer2(self.convnet_2.end_relu(x_new_1))

        x_new_2 = 0.5*level2_x_1 + 0.5*level2_x_2

        level3_x_1 = self.convnet_1.layer3(self.convnet_1.end_relu(x_new_2))
        level3_x_2 = self.convnet_2.layer3(self.convnet_2.end_relu(x_new_2))

        x_new_3 = 0.5*level3_x_1 + 0.5*level3_x_2

        level4_x_1 = self.convnet_1.layer4(self.convnet_1.end_relu(x_new_3))
        level4_x_2 = self.convnet_2.layer4(self.convnet_2.end_relu(x_new_3))

        x_new_4 = 0.5*level4_x_1 + 0.5*level4_x_2

        # self.convnet_2.final_layer
        raw_features = self.end_features_img_net(x_new_4)
        features = self.end_features_img_net(F.relu(x_new_4, inplace=False))
        return {
            "raw_features": raw_features,
            "features": features,
            "attention": [x_new_1, x_new_2, x_new_3, x_new_4]
        }

    def forward_convnet_cifar100(self, x):
        x_1 = self.convnet_1.conv_1_3x3(x)
        x_1 = F.relu(self.convnet_1.bn_1(x_1), inplace=True)
        feats_s1_1, x_1 = self.convnet_1.stage_1(x_1)

        x_2 = self.convnet_2.conv_1_3x3(x)
        x_2 = F.relu(self.convnet_2.bn_1(x_2), inplace=True)
        feats_s1_2, x_2 = self.convnet_2.stage_1(x_2)

        x_new = 0.5*x_1 + 0.5*x_2
        feats_s1 = 0.5*feats_s1_1[-1] + 0.5*feats_s1_2[-1]

        feats_s2_1, x_1 = self.convnet_1.stage_2(x_new)
        feats_s2_2, x_2 = self.convnet_2.stage_2(x_new)
        x_new = 0.5*x_1 + 0.5*x_2
        feats_s2 = 0.5*feats_s2_1[-1] + 0.5*feats_s2_2[-1]

        feats_s3_1, x_1 = self.convnet_1.stage_3(x_new)
        feats_s3_2, x_2 = self.convnet_2.stage_3(x_new)
        x_new = 0.5*x_1 + 0.5*x_2
        feats_s3 = 0.5*feats_s3_1[-1] + 0.5*feats_s3_2[-1]

        x_1 = self.convnet_1.stage_4(x_new)
        x_2 = self.convnet_2.stage_4(x_new)
        x_new = 0.5*x_1 + 0.5*x_2

        raw_features = self.end_features(x_new)
        features = self.end_features(F.relu(x_new, inplace=False))
        attentions = [feats_s1, feats_s2, feats_s3, x_new]

        return {"raw_features": raw_features, "features": features, "attention": attentions}

    def end_features(self, x):
        x = self.convnet_1.pool(x)
        x = x.view(x.size(0), -1)

        return x

    def end_features_img_net(self, x):
        x = self.convnet_1.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(
        self, x, rotation=False, index=None, features_processing=None, additional_features=None
    ):
        if hasattr(self,
                   "word_embeddings") and self.word_embeddings is not None and isinstance(x, list):
            words = x[1]
            x = x[0]
        else:
            words = None
        outputs = self.forward_convnet(x)
        if words is not None: 
            outputs["word_embeddings"] = self.word_embeddings(words)

        if hasattr(self, "classifier_no_act") and self.classifier_no_act:
            selected_features = outputs["raw_features"]
        else:
            selected_features = outputs["features"]

        if features_processing is not None:
            selected_features = features_processing.fit_transform(selected_features)

        if rotation:
            outputs["rotations"] = self.rotations_predictor(outputs["features"])
            nb_inputs = len(x) // 4
        else:
            if additional_features is not None:
                clf_outputs = self.classifier(
                    torch.cat((selected_features, additional_features), 0)
                )
            else:
                clf_outputs = self.classifier(selected_features)
            outputs.update(clf_outputs)

        if hasattr(self, "gradcam_hook") and self.gradcam_hook:
            outputs["gradcam_gradients"] = self._gradcam_gradients
            outputs["gradcam_activations"] = self._gradcam_activations

        return outputs

    def post_process(self, x):
        if self.post_processor is None:
            return x
        return self.post_processor(x)

    @property
    def features_dim(self):
        return self.convnet_1.out_dim

    def add_classes(self, n_classes):
        self.classifier.add_classes(n_classes)

    def add_imprinted_classes(self, class_indexes, inc_dataset, **kwargs):
        if hasattr(self.classifier, "add_imprinted_classes"):
            self.classifier.add_imprinted_classes(class_indexes, inc_dataset, self, **kwargs)

    def add_custom_weights(self, weights, **kwargs):
        self.classifier.add_custom_weights(weights, **kwargs)

    def extract(self, x):
        outputs = self.forward_convnet(x)
        if self.extract_no_act:
            try:
                return outputs["raw_features"]
            except:
                import pdb
                pdb.set_trace()
        return outputs["features"]

    def freeze(self, trainable=False, model="all"):
        if model == "all":
            model = self
        elif model == "classifier":
            model = self.classifier
        else:
            assert False, model

        if not isinstance(model, nn.Module):
            return self

        for param in model.parameters():
            param.requires_grad = trainable

        if not trainable:
            model.eval()
        else:
            model.train()

        return self

    def get_group_parameters(self):
        groups = {"convnet_1": self.convnet_1.parameters(), "convnet_2": self.convnet_2.parameters()}

        if isinstance(self.post_processor, FactorScalar):
            groups["postprocessing"] = self.post_processor.parameters()
        if hasattr(self.classifier, "new_weights"):
            groups["new_weights"] = self.classifier.new_weights
        if hasattr(self.classifier, "old_weights"):
            groups["old_weights"] = self.classifier.old_weights
        if self.rotations_predictor:
            groups["rotnet"] = self.rotations_predictor.parameters()
        if hasattr(self.convnet_1, "last_block"):
            groups["last_block_1"] = self.convnet_1.last_block.parameters()
        if hasattr(self.convnet_2, "last_block"):
            groups["last_block_2"] = self.convnet_2.last_block.parameters()
        if hasattr(self.classifier, "_negative_weights"
                  ) and isinstance(self.classifier._negative_weights, nn.Parameter):
            groups["neg_weights"] = self.classifier._negative_weights
        if self.domain_classifier is not None:
            groups["domain_clf"] = self.domain_classifier.parameters()

        return groups

    def copy(self):
        return copy.deepcopy(self)

    @property
    def n_classes(self):
        return self.classifier.n_classes

    def unset_gradcam_hook(self):
        self._hooks[0].remove()
        self._hooks[1].remove()
        self._hooks[0] = None
        self._hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)

    def create_domain_classifier(self):
        self.domain_classifier = DomainClassifier(self.convnet.out_dim, device=self.device)
        return self.domain_classifier

    def del_domain_classifier(self):
        self.domain_classifier = None
