-- Them "\hline" sau moi dong du lieu trong bang (khong chi duoi header),
-- khi xuat ra LaTeX.
--
-- Vi sao lam the nay thay vi dinh nghia lai "\\" ngay trong longtable:
-- longtable tu dung "\\" cho ca viec dung phan dau/chan bang, nen dinh
-- nghia lai no se pha co che noi bo cua longtable (loi "Misplaced
-- \noalign"). Cach an toan: de pandoc tu render bang nhu binh thuong
-- (giu nguyen toan bo logic can chinh cot/in dam/xuong dong cua no), roi
-- chi CHEN THANG chuoi "\hline" vao ngay sau moi dong du lieu bang xu ly
-- van ban thuan tren ket qua da render - khong dung macro TeX nao ca.

function Table(tbl)
  if not FORMAT:match("latex") then
    return nil
  end

  local latex = pandoc.write(pandoc.Pandoc({ tbl }), "latex")

  local patched = latex:gsub(
    "(\\endlastfoot\n)(.-)(\n\\end{longtable%*?})",
    function(head, body, tail)
      local lines = {}
      for line in (body .. "\n"):gmatch("(.-)\n") do
        table.insert(lines, line)
        if line:match("\\\\%s*$") then
          table.insert(lines, "\\hline")
        end
      end
      return head .. table.concat(lines, "\n") .. tail
    end
  )

  return pandoc.RawBlock("latex", patched)
end
