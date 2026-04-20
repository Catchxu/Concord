import torch

from src.data import GeneVocab
from src.models import ExpressionTokenizer, GeneTokenizer


def test_gene_tokenizer_accepts_partial_gene2vec_init():
    gene_vocab = GeneVocab.from_gene_names(
        ["g1", "g2", "g3"],
        gene2vec={"g2": [0.1, 0.2, 0.3, 0.4]},
    )
    tokenizer = GeneTokenizer(gene_vocab=gene_vocab, embed_dim=4)
    token_id = gene_vocab.token_id_from_gene_name("g2")
    expected = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=tokenizer.embedding.weight.dtype)
    assert torch.allclose(tokenizer.embedding.weight[token_id], expected)


def test_expression_tokenizer_bins_deterministically():
    tokenizer = ExpressionTokenizer(embed_dim=8, num_bins=4, max_log1p_value=4.0)
    expression_values = torch.tensor([[0.0, 1.0, 3.0, 10.0]], dtype=torch.float32)
    padding_mask = torch.tensor([[False, False, False, False]], dtype=torch.bool)
    bin_ids_a = tokenizer.build_input_ids(expression_values, padding_mask)
    bin_ids_b = tokenizer.build_input_ids(expression_values, padding_mask)
    assert torch.equal(bin_ids_a, bin_ids_b)
    assert bin_ids_a[0, 0].item() == tokenizer.cls_token_id
    assert bin_ids_a[0, 1].item() >= tokenizer.bin_offset
