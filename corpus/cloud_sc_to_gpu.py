from pathlib import Path

import dace

from corpus.offload_cloudsc_to_gpu import offload_cloudsc_to_gpu

CORPUS_DIR = Path(__file__).resolve().parent

__all__ = ["offload_cloudsc_to_gpu"]


def main() -> None:

    sdfg = dace.SDFG.from_file(CORPUS_DIR / "optimized_cloudsc.sdfg")
    offload_cloudsc_to_gpu(sdfg)


if __name__ == "__main__":
    main()
