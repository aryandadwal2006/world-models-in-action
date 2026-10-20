# Offset top-level section numbers for standalone chapter builds.
# The :leveloffset: 1 in the AsciiDoc source makes its chapter title a level-1
# section when compiled on its own or included under a book title.
require 'asciidoctor/extensions' unless RUBY_ENGINE == 'opal'

Asciidoctor::Extensions.register do
  treeprocessor do
    process do |document|
      offset = (document.attr 'sectnumoffset', 0).to_i

      if offset != 0
        document.find_by(context: :section) do |section|
          section.level == 1 && section.numbered
        end.each do |section|
          if offset > 0
            offset.times { section.numeral = section.numeral.next } rescue (section.number += offset)
          else
            offset.abs.times { section.numeral = section.numeral.to_i.pred.to_s } rescue (section.number += offset)
          end
        end
      end

      nil
    end
  end
end
