//! Turns the text file into pages of lines that fit the text area.

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// What happens to a line that is wider than the text area.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Wrap {
    /// Continue it on the next row, breaking between words where possible
    Word,
    /// Cut it off at the right edge
    None,
}

/// One row of text on a page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Line {
    pub text: String,
    /// Where the row comes from: the number of the line in the file, and which of the rows that line was
    /// wrapped into this is. Both count from 0 and together they order the rows of a text.
    pub at: (usize, usize),
}

/// The rows of one page, at least one.
pub type Page = Vec<Line>;

/// A tab moves on to the next multiple of this many columns.
const TAB_STOP: usize = 4;

/// Splits `text` into pages of at most `height` rows of at most `width` columns. A form feed character
/// starts a new page. Empty rows are kept, except where they would begin or end a page.
pub fn paginate(text: &str, width: usize, height: usize, wrap: Wrap) -> Vec<Page> {
    let mut pages = Vec::new();
    if width == 0 || height == 0 {
        return pages;
    }
    let mut number = 0;
    for section in text.split('\x0c') {
        let mut page = Page::new();
        for line in section.lines() {
            let line = clean(line);
            let rows = match wrap {
                Wrap::Word => wrap_words(&line, width),
                Wrap::None => vec![fit(&line, width).to_owned()],
            };
            for (piece, text) in rows.into_iter().enumerate() {
                if page.is_empty() && text.is_empty() {
                    continue;
                }
                page.push(Line {
                    text,
                    at: (number, piece),
                });
                if page.len() == height {
                    close(&mut page, &mut pages);
                }
            }
            number += 1;
        }
        close(&mut page, &mut pages);
    }
    pages
}

/// Moves the rows collected in `page` to `pages`, without the empty rows at their end.
fn close(page: &mut Page, pages: &mut Vec<Page>) {
    while page.last().is_some_and(|line| line.text.is_empty()) {
        page.pop();
    }
    if !page.is_empty() {
        pages.push(std::mem::take(page));
    }
}

/// The line as it is shown: tabs as spaces up to the next tab stop, no control characters, no spaces at
/// the end.
fn clean(line: &str) -> String {
    let mut cleaned = String::with_capacity(line.len());
    let mut column = 0;
    for c in line.chars() {
        if c == '\t' {
            let spaces = TAB_STOP - column % TAB_STOP;
            cleaned.extend(std::iter::repeat_n(' ', spaces));
            column += spaces;
        } else if !c.is_control() {
            cleaned.push(c);
            column += c.width().unwrap_or(0);
        }
    }
    cleaned.truncate(cleaned.trim_end().len());
    cleaned
}

/// The longest beginning of `text` that fits into `width` columns.
pub fn fit(text: &str, width: usize) -> &str {
    let mut used = 0;
    for (index, c) in text.char_indices() {
        used += c.width().unwrap_or(0);
        if used > width {
            return &text[..index];
        }
    }
    text
}

/// The line as rows of at most `width` columns, broken at spaces; a word that is wider than a row on its
/// own is broken wherever the row ends. The first row keeps the indentation of the line, the others start
/// with a word. An empty line is one empty row.
fn wrap_words(line: &str, width: usize) -> Vec<String> {
    let mut rows = Vec::new();
    let mut row = String::new();
    // the spaces since the last word, which are only shown between two words of the same row
    let mut gap = "";
    for token in tokens(line) {
        if token.starts_with(' ') {
            let continues_a_line = row.is_empty() && !rows.is_empty();
            if !continues_a_line {
                gap = token;
            }
            continue;
        }
        let mut word = token;
        if row.width() + gap.width() + word.width() <= width {
            row.push_str(gap);
        } else {
            if !row.is_empty() {
                rows.push(std::mem::take(&mut row));
            }
            while word.width() > width {
                let mut head = fit(word, width);
                if head.is_empty() {
                    // a character wider than the whole row still has to go somewhere
                    head = &word[..word.chars().next().map_or(0, char::len_utf8)];
                }
                rows.push(head.to_owned());
                word = &word[head.len()..];
            }
        }
        row.push_str(word);
        gap = "";
    }
    if !row.is_empty() || rows.is_empty() {
        rows.push(row);
    }
    rows
}

/// The runs of spaces and the runs of other characters that `line` consists of, in turn.
fn tokens(line: &str) -> impl Iterator<Item = &str> {
    let mut rest = line;
    std::iter::from_fn(move || {
        let spaces = rest.chars().next()? == ' ';
        let end = rest.find(|c: char| (c == ' ') != spaces).unwrap_or(rest.len());
        let (token, tail) = rest.split_at(end);
        rest = tail;
        Some(token)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(pages: &[Page]) -> Vec<Vec<&str>> {
        pages
            .iter()
            .map(|page| page.iter().map(|line| line.text.as_str()).collect())
            .collect()
    }

    #[test]
    fn long_lines_are_broken_between_words() {
        assert_eq!(
            wrap_words("the quick brown fox jumps", 10),
            ["the quick", "brown fox", "jumps"]
        );
        // exactly as wide as the row
        assert_eq!(wrap_words("abcde fghij", 5), ["abcde", "fghij"]);
        assert_eq!(wrap_words("", 5), [""]);
    }

    #[test]
    fn spacing_inside_a_row_is_kept_and_dropped_at_a_break() {
        assert_eq!(wrap_words("  a   b", 10), ["  a   b"]);
        assert_eq!(
            wrap_words("  indented text here", 12),
            ["  indented", "text here"]
        );
        assert_eq!(wrap_words("one     two", 5), ["one", "two"]);
    }

    #[test]
    fn a_word_wider_than_the_row_is_broken_inside() {
        assert_eq!(
            wrap_words("a 0123456789abc b", 5),
            ["a", "01234", "56789", "abc b"]
        );
        // two columns per character
        assert_eq!(wrap_words("漢字漢字漢", 4), ["漢字", "漢字", "漢"]);
        assert_eq!(wrap_words("漢字", 1), ["漢", "字"]);
    }

    #[test]
    fn tabs_and_control_characters_are_resolved() {
        assert_eq!(clean("a\tb\t\tc"), "a   b       c");
        assert_eq!(clean("bell\x07 and escape\x1b[0m  \r"), "bell and escape[0m");
    }

    #[test]
    fn unwrapped_lines_are_cut_off() {
        let pages = paginate("0123456789\nab", 4, 5, Wrap::None);
        assert_eq!(texts(&pages), [["0123", "ab"]]);
    }

    #[test]
    fn pages_fill_up_and_remember_where_their_rows_come_from() {
        let pages = paginate("one two three\nfour\nfive\n", 9, 2, Wrap::Word);
        assert_eq!(texts(&pages), [vec!["one two", "three"], vec!["four", "five"]]);
        let origins = |page: &Page| page.iter().map(|line| line.at).collect::<Vec<_>>();
        assert_eq!(origins(&pages[0]), [(0, 0), (0, 1)]);
        assert_eq!(origins(&pages[1]), [(1, 0), (2, 0)]);
    }

    #[test]
    fn empty_rows_stay_inside_a_page_only() {
        let pages = paginate("\n\na\n\nb\n\n\nc\n\n", 20, 3, Wrap::Word);
        assert_eq!(texts(&pages), [vec!["a", "", "b"], vec!["c"]]);
    }

    #[test]
    fn form_feed_starts_a_page() {
        let pages = paginate("a\nb\n\x0cc\n\x0c\x0c", 20, 10, Wrap::Word);
        assert_eq!(texts(&pages), [vec!["a", "b"], vec!["c"]]);
        assert!(paginate("", 20, 10, Wrap::Word).is_empty());
    }
}
