from typing import Union, Sequence
from pathlib import Path
from datetime import datetime


def drop_cols(file: Union[Path, str],
              file_out: Union[Path, str],
              cols: Sequence[str] = []):
    "Copy csv file without selected columns"
    # helper function for applying mask to a csv file line
    def filter_line(line: str, mask: list):
        line_lst = line.rstrip().split(",")
        filtered = [col for col, drop in zip(line_lst, mask) if not drop]
        return ",".join(filtered) + "\n"

    with open(file, "r") as fin:
        # get boolean mask from header
        header = next(fin)
        header_lst = header.rstrip().split(',')
        mask = [col in cols for col in header_lst]

        # apply mask to every line and export results
        with open(file_out, "w") as fout:
            fout.write(filter_line(header, mask))
            for line in fin:
                fout.write(filter_line(line, mask))


def str_to_date(s: str) -> datetime.date:
    return datetime.fromisoformat(s).date()


def date_split(file: Union[Path, str],
               output_dir: Union[Path, str],
               test_date: str = "2021-10-02",
               date_col: int = 0):
    """
    Split csv file by putting all entries with `test_date` date into test
    dataset. Creates `train.csv` and `test.csv` in `output_dir`.
    """
    output_dir = Path(output_dir)
    assert output_dir.exists()
    test_date = str_to_date(test_date)

    with open(file, "r") as infile:
        # output file objects
        f_train = open(output_dir / "train.csv", "w")
        f_test = open(output_dir / "test.csv", "w")

        # write header
        header = next(infile)
        f_train.write(header)
        f_test.write(header)

        # copy data to one of the files line by line
        for line in infile:
            date_str = line.split(",")[date_col]
            date = str_to_date(date_str)
            fout = f_test if date == test_date else f_train
            fout.write(line)

        f_train.close()
        f_test.close()
