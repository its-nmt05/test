from transformers import EncodecModel


def load_nac():
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")   
    return model.encoder, model.decoder, model.quantizer

load_nac()